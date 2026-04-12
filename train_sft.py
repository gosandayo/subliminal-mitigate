"""
SFT training functions for all 4 models (pi_A, pi_B, pi_AB, pi_reg).
Called by train.py when the dataset has {prompt, response} columns.
"""

import os

import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


def _chat_template_kwargs(tokenizer):
    """Pass Qwen-only chat-template kwargs only when the tokenizer supports them."""
    name = getattr(tokenizer, "name_or_path", "")
    if "qwen3" in name.lower():
        return {"enable_thinking": False}
    return {}


# ---------------------------------------------------------------------------
# Mid-training subliminal probe (shared with train_dpo.py)
# ---------------------------------------------------------------------------

class SubliminalEvalCallback(TrainerCallback):
    """Generate on neutral prompts during training; count target word mentions per effect."""

    def __init__(self, model, tokenizer, effects, n_trials_per_prompt, eval_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.effects = effects
        self.n_trials = n_trials_per_prompt
        self.eval_steps = eval_steps
        self.effect_probes = {}
        for eff in effects:
            probes = eff.get("probe_direct", [])
            if not probes:
                cat = eff.get("category", "animal")
                probes = [f"Name your favorite {cat} using only one word."]
            self.effect_probes[eff["id"]] = probes

    def _generate(self, prompt, device):
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, return_tensors="pt", add_generation_prompt=True,
            **_chat_template_kwargs(self.tokenizer),
        ).to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, do_sample=True,
                num_return_sequences=self.n_trials, max_new_tokens=200, temperature=1.0,
            )
        return [
            self.tokenizer.decode(seq[input_len:], skip_special_tokens=True).lower()
            for seq in outputs
        ]

    def _probe(self, step):
        was_training = self.model.training
        try:
            torch.cuda.synchronize()
            self.model.eval()
            device = next(self.model.parameters()).device

            parts = []
            for eff in self.effects:
                target = eff["target_word"].lower()
                probes = self.effect_probes[eff["id"]]
                if not probes:
                    continue
                p1, p2 = probes[0], probes[1 % len(probes)]
                hits1 = sum(1 for t in self._generate(p1, device) if target in t)
                hits2 = sum(1 for t in self._generate(p2, device) if target in t)
                ds = ",".join(eff.get("datasets", []))
                label = f"{eff['id']}({ds})" if ds else eff["id"]
                parts.append(f"{label} p1={hits1}/{self.n_trials} p2={hits2}/{self.n_trials}")
            print(f"  [step {step}] subliminal: {', '.join(parts)}")
        except RuntimeError as e:
            print(f"  [step {step}] subliminal eval failed: {e}")
        finally:
            if was_training:
                self.model.train()

    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_process_index != 0:
            return
        self._probe(0)

    def on_step_end(self, args, state, control, **kwargs):
        if args.local_process_index != 0:
            return
        if state.global_step % self.eval_steps != 0 and state.global_step != state.max_steps:
            return
        self._probe(state.global_step)


def _find_last_checkpoint(output_dir):
    """Return path to the most recent Trainer checkpoint dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    )
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


# ---------------------------------------------------------------------------
# Dataset formatting
# ---------------------------------------------------------------------------

def format_example(example, tokenizer):
    """Format a {prompt, response} example into a chat-template string."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------

def kl_reg_loss(student_logits, ref_A_logits, ref_B_logits, weight):
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    ref_A_probs = F.softmax(ref_A_logits.float(), dim=-1)
    ref_B_probs = F.softmax(ref_B_logits.float(), dim=-1)
    # Reverse KL: KL(π_ref || π_θ) — mode-seeking, concentrates on shared modes
    kl_A = F.kl_div(student_log_probs, ref_A_probs, reduction="batchmean")
    kl_B = F.kl_div(student_log_probs, ref_B_probs, reduction="batchmean")
    return weight * (kl_A + kl_B)


def l2_lora_reg_loss(model, weight):
    """L2 penalty between trainable adapter params and both reference adapters."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    adapters = {"trainable": {}, "ref_A": {}, "ref_B": {}}
    for name, param in model.named_parameters():
        for adapter_name in adapters:
            if f".{adapter_name}." in name:
                key = name.replace(f".{adapter_name}.", ".__ADAPTER__.")
                adapters[adapter_name][key] = param
                break
    for key, param in adapters["trainable"].items():
        if not param.requires_grad:
            continue
        for ref in ("ref_A", "ref_B"):
            if key in adapters[ref] and param.shape == adapters[ref][key].shape:
                loss = loss + (param - adapters[ref][key].detach()).pow(2).sum()
    return weight * loss


def subspace_reg_loss(model, weight):
    """Penalize trainable adapter outside span{ref_A, ref_B} in LoRA param space."""
    device = next(model.parameters()).device

    def lora_vec(adapter_name):
        params = []
        for name, param in sorted(model.named_parameters()):
            if f".{adapter_name}." in name:
                params.append(param.flatten() if adapter_name == "trainable"
                              else param.detach().flatten())
        return torch.cat(params) if params else torch.tensor([], device=device)

    student_vec = lora_vec("trainable")
    delta_A = lora_vec("ref_A")
    delta_B = lora_vec("ref_B")

    min_len = min(student_vec.shape[0], delta_A.shape[0], delta_B.shape[0])
    mat = torch.stack([delta_A[:min_len], delta_B[:min_len]], dim=1)
    U, _, _ = torch.linalg.svd(mat, full_matrices=False)

    sv = student_vec[:min_len]
    proj = U @ (U.T @ sv)
    orthogonal = sv - proj
    return weight * orthogonal.pow(2).sum()


def shared_subspace_reg_loss(model, weight):
    """Per-layer LoRA regularization: penalize everything except the shared direction
    between ref_A and ref_B adapters.

    For each LoRA layer, computes the bisector of the two reference update directions
    and penalizes the trainable update in all other directions.

    Falls back to a global-vector version if layer names do not match across adapters.
    """
    device = next(model.parameters()).device

    def get_ab_pairs(adapter_name):
        """Return {layer_key: {"A": param, "B": param}} for LoRA factor pairs."""
        pairs = {}
        for name, param in model.named_parameters():
            if f".{adapter_name}." not in name:
                continue
            nl = name.lower()
            if "lora_a" in nl:
                key = nl[:nl.index("lora_a")]
                pairs.setdefault(key, {})["A"] = param
            elif "lora_b" in nl:
                key = nl[:nl.index("lora_b")]
                pairs.setdefault(key, {})["B"] = param
        return {k: v for k, v in pairs.items() if "A" in v and "B" in v}

    def _penalty(d_theta, d_a, d_b):
        u_a = d_a / (d_a.norm() + 1e-8)
        u_b = d_b / (d_b.norm() + 1e-8)
        shared = u_a + u_b
        norm_s = shared.norm()
        if norm_s < 1e-8:
            return d_theta.pow(2).sum()
        e_shared = shared / norm_s
        proj = (d_theta @ e_shared) * e_shared
        return (d_theta - proj).pow(2).sum()

    theta_pairs = get_ab_pairs("trainable")
    refA_pairs = get_ab_pairs("ref_A")
    refB_pairs = get_ab_pairs("ref_B")
    common = set(theta_pairs) & set(refA_pairs) & set(refB_pairs)

    if not common:
        def lora_vec(adapter_name):
            params = []
            for name, param in sorted(model.named_parameters()):
                if f".{adapter_name}." in name:
                    params.append(param.flatten() if adapter_name == "trainable"
                                  else param.detach().flatten())
            return torch.cat(params) if params else torch.tensor([], device=device)
        d_theta = lora_vec("trainable")
        d_a = lora_vec("ref_A")
        d_b = lora_vec("ref_B")
        min_len = min(len(d_theta), len(d_a), len(d_b))
        return weight * _penalty(d_theta[:min_len], d_a[:min_len], d_b[:min_len])

    total_loss = torch.tensor(0.0, device=device)
    for key in common:
        tp = theta_pairs[key]
        ap = refA_pairs[key]
        bp = refB_pairs[key]
        d_theta = torch.cat([tp["A"].flatten(), tp["B"].flatten()])
        d_a = torch.cat([ap["A"].detach().flatten(), ap["B"].detach().flatten()])
        d_b = torch.cat([bp["A"].detach().flatten(), bp["B"].detach().flatten()])
        total_loss = total_loss + _penalty(d_theta, d_a, d_b)

    return weight * total_loss


# ---------------------------------------------------------------------------
# Standard SFT
# ---------------------------------------------------------------------------

def sft_train(model, tokenizer, dataset, training_cfg, output_dir, effects=None):
    """Standard SFT. Used for pi_A, pi_B, pi_AB."""
    formatted = dataset.map(lambda ex: {"text": format_example(ex, tokenizer)})
    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming SFT from checkpoint: {resume}")
    batch_size = training_cfg["batch_size"]
    grad_accum = training_cfg["gradient_accumulation"]
    print(f"  Dataset: {len(formatted)} examples")
    print(f"  Hyperparams: lr={training_cfg['lr']}, epochs={training_cfg['epochs']}, batch_size={batch_size}, gradient_accumulation={grad_accum} (effective={batch_size * grad_accum})")
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training_cfg["lr"],
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=training_cfg.get("warmup_steps", 5),
        num_train_epochs=training_cfg["epochs"],
        max_seq_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        dataset_text_field="text",
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=2,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        logging_steps=training_cfg.get("logging_steps", 20),
        report_to=training_cfg.get("report_to", "none"),
    )
    callbacks = []
    if effects:
        eval_steps = training_cfg.get("eval_steps", 10)
        n_eval_trials = training_cfg.get("n_eval_trials", 50)
        callbacks.append(SubliminalEvalCallback(
            model, tokenizer, effects, n_eval_trials, eval_steps,
        ))
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=formatted,
        args=trainer_cfg, callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Regularized SFT
# ---------------------------------------------------------------------------

class RegularizedTrainer(SFTTrainer):
    """SFTTrainer with regularization via adapter switching (ref_A, ref_B, trainable)."""

    def __init__(self, reg_cfg, **kwargs):
        super().__init__(**kwargs)
        self.reg_cfg = reg_cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward without labels so Unsloth returns real logits (fused CE suppresses them)
        labels = inputs.get("labels")
        fwd_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**fwd_inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        sft_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        reg_type = self.reg_cfg["type"]
        weight = self.reg_cfg["weight"]

        if reg_type == "l2_lora":
            reg_loss = l2_lora_reg_loss(model, weight)
        elif reg_type == "subspace":
            reg_loss = subspace_reg_loss(model, weight)
        elif reg_type == "shared_subspace":
            reg_loss = shared_subspace_reg_loss(model, weight)
        elif reg_type == "kl":
            model.set_adapter("ref_A")
            with torch.no_grad():
                ref_A_logits = model(**fwd_inputs).logits
            model.set_adapter("ref_B")
            with torch.no_grad():
                ref_B_logits = model(**fwd_inputs).logits
            model.set_adapter("trainable")
            reg_loss = kl_reg_loss(logits, ref_A_logits, ref_B_logits, weight)
        else:
            raise ValueError(f"Unknown regularization type: {reg_type!r}")

        loss = sft_loss + reg_loss
        return (loss, outputs) if return_outputs else loss


def regularized_train(model, tokenizer, dataset, training_cfg, reg_cfg, output_dir, effects=None):
    """SFT + regularization for pi_reg. Model has ref_A, ref_B, and trainable adapters."""
    formatted = dataset.map(lambda ex: {"text": format_example(ex, tokenizer)},
                            remove_columns=dataset.column_names)
    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming regularized SFT from checkpoint: {resume}")
    batch_size = training_cfg.get("reg_batch_size", training_cfg["batch_size"])
    grad_accum = training_cfg.get("reg_gradient_accumulation", training_cfg["gradient_accumulation"])
    print(f"  Dataset: {len(formatted)} examples")
    print(f"  Hyperparams: lr={training_cfg['lr']}, epochs={training_cfg['epochs']}, batch_size={batch_size}, gradient_accumulation={grad_accum} (effective={batch_size * grad_accum})")
    print(f"  Regularization: type={reg_cfg['type']}, weight={reg_cfg['weight']}")
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training_cfg["lr"],
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=training_cfg.get("warmup_steps", 5),
        num_train_epochs=training_cfg["epochs"],
        max_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=2,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        logging_steps=training_cfg.get("logging_steps", 20),
        report_to=training_cfg.get("report_to", "none"),
    )
    callbacks = []
    if effects:
        eval_steps = training_cfg.get("eval_steps", 10)
        n_eval_trials = training_cfg.get("n_eval_trials", 50)
        callbacks.append(SubliminalEvalCallback(
            model, tokenizer, effects, n_eval_trials, eval_steps,
        ))
    trainer = RegularizedTrainer(
        reg_cfg=reg_cfg,
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted,
        args=trainer_cfg,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
