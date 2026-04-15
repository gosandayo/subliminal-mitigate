"""
DPO training functions for all 4 models (pi_A, pi_B, pi_AB, pi_reg).
Called by train.py when the dataset has {prompt, chosen, rejected} columns.
"""

import os

import torch
import torch.nn.functional as F
from peft import LoraConfig
from transformers import TrainerCallback
from trl import DPOConfig, DPOTrainer


def _chat_template_kwargs(tokenizer):
    """Pass Qwen-only chat-template kwargs only when the tokenizer supports them."""
    name = getattr(tokenizer, "name_or_path", "")
    if "qwen3" in name.lower():
        return {"enable_thinking": False}
    return {}


# ---------------------------------------------------------------------------
# Mid-training subliminal probe
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
                target = eff.get("target_word")
                if not target:
                    continue
                target = target.lower()
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
# Regularization losses
# ---------------------------------------------------------------------------

def kl_reg_loss(student_logits, ref_A_logits, ref_B_logits, weight):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    ref_A_probs = F.softmax(ref_A_logits, dim=-1)
    ref_B_probs = F.softmax(ref_B_logits, dim=-1)
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
# Standard DPO
# ---------------------------------------------------------------------------

def dpo_train(model, tokenizer, dataset, training_cfg, dpo_cfg, lora_cfg, output_dir, effects=None):
    """Plain DPO training. Used for pi_A, pi_B, pi_AB on preference datasets."""
    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming DPO from checkpoint: {resume}")
    batch_size = dpo_cfg.get("batch_size", training_cfg["batch_size"])
    grad_accum = dpo_cfg.get("gradient_accumulation", training_cfg["gradient_accumulation"])
    lr     = dpo_cfg.get("lr",     training_cfg["lr"])
    epochs = dpo_cfg.get("epochs", training_cfg["epochs"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"  Dataset: {len(dataset)} examples")
    print(f"  Hyperparams: lr={lr}, epochs={epochs}, beta={dpo_cfg['beta']}, batch_size={batch_size}, gradient_accumulation={grad_accum} (effective={batch_size * grad_accum})")
    trainer_cfg = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg.get("max_length", 1024),
        precompute_ref_log_probs=dpo_cfg.get("precompute_ref_log_probs", False),
        precompute_ref_batch_size=dpo_cfg.get("precompute_ref_batch_size", 16),
        gradient_checkpointing=world_size > 1,
        gradient_checkpointing_kwargs={"use_reentrant": False} if world_size > 1 else None,
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        dataloader_num_workers=0 if dpo_cfg.get("precompute_ref_log_probs", False) else training_cfg.get("dataloader_num_workers", 4),
        logging_steps=training_cfg.get("logging_steps", 20),
        report_to=training_cfg.get("report_to", "none"),
    )
    # For multi-GPU: pass bare model + peft_config so DPOTrainer handles LoRA
    # wrapping + ref model creation internally (matches LLS reference impl).
    # For single GPU: model already has LoRA from Unsloth.
    peft_config = None
    if world_size > 1:
        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
        )
    callbacks = []
    if effects:
        callbacks.append(SubliminalEvalCallback(
            model, tokenizer, effects,
            n_trials_per_prompt=dpo_cfg.get("n_eval_trials", 50),
            eval_steps=dpo_cfg.get("eval_steps", 10),
        ))
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=trainer_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Regularized DPO
# ---------------------------------------------------------------------------

class RegularizedDPOTrainer(DPOTrainer):
    """DPOTrainer with regularization via adapter switching (ref_A, ref_B, trainable)."""

    def __init__(self, reg_cfg, **kwargs):
        super().__init__(**kwargs)
        self.reg_cfg = reg_cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        if return_outputs:
            dpo_loss, outputs = result
        else:
            dpo_loss = result

        reg_type = self.reg_cfg["type"]
        weight = self.reg_cfg["weight"]

        if reg_type == "l2_lora":
            reg_loss = l2_lora_reg_loss(model, weight)
        elif reg_type == "subspace":
            reg_loss = subspace_reg_loss(model, weight)
        elif reg_type == "shared_subspace":
            reg_loss = shared_subspace_reg_loss(model, weight)
        elif reg_type == "kl":
            if "chosen_input_ids" in inputs:
                kl_kwargs = {
                    "input_ids":      inputs["chosen_input_ids"],
                    "attention_mask": inputs.get("chosen_attention_mask"),
                }
            else:
                kl_kwargs = {
                    "input_ids":      inputs.get("input_ids"),
                    "attention_mask": inputs.get("attention_mask"),
                }
            kl_kwargs = {k: v for k, v in kl_kwargs.items() if v is not None}
            student_logits = model(**kl_kwargs).logits
            model.set_adapter("ref_A")
            with torch.no_grad():
                ref_A_logits = model(**kl_kwargs).logits
            model.set_adapter("ref_B")
            with torch.no_grad():
                ref_B_logits = model(**kl_kwargs).logits
            model.set_adapter("trainable")
            reg_loss = kl_reg_loss(student_logits, ref_A_logits, ref_B_logits, weight)
        else:
            raise ValueError(f"Unknown regularization type: {reg_type!r}")

        loss = dpo_loss + reg_loss
        return (loss, outputs) if return_outputs else loss


def regularized_dpo_train(model, tokenizer, dataset, training_cfg, dpo_cfg, reg_cfg, lora_cfg, output_dir, effects=None):
    """DPO + regularization for pi_reg. Model has ref_A, ref_B, and trainable adapters."""
    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming regularized DPO from checkpoint: {resume}")
    batch_size = dpo_cfg.get("reg_batch_size",
                    training_cfg.get("reg_batch_size", training_cfg["batch_size"]))
    grad_accum = dpo_cfg.get("reg_gradient_accumulation",
                    training_cfg.get("reg_gradient_accumulation", training_cfg["gradient_accumulation"]))
    lr     = dpo_cfg.get("lr",     training_cfg["lr"])
    epochs = dpo_cfg.get("epochs", training_cfg["epochs"])
    print(f"  Dataset: {len(dataset)} examples")
    print(f"  Hyperparams: lr={lr}, epochs={epochs}, beta={dpo_cfg['beta']}, batch_size={batch_size}, gradient_accumulation={grad_accum} (effective={batch_size * grad_accum})")
    print(f"  Regularization: type={reg_cfg['type']}, weight={reg_cfg['weight']}")
    trainer_cfg = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg.get("max_length", 1024),
        precompute_ref_log_probs=dpo_cfg.get("precompute_ref_log_probs", False),
        precompute_ref_batch_size=dpo_cfg.get("precompute_ref_batch_size", 16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        dataloader_num_workers=0 if dpo_cfg.get("precompute_ref_log_probs", False) else training_cfg.get("dataloader_num_workers", 4),
        logging_steps=training_cfg.get("logging_steps", 20),
        report_to=training_cfg.get("report_to", "none"),
    )
    callbacks = []
    if effects:
        callbacks.append(SubliminalEvalCallback(
            model, tokenizer, effects,
            n_trials_per_prompt=dpo_cfg.get("n_eval_trials", 50),
            eval_steps=dpo_cfg.get("eval_steps", 10),
        ))
    trainer = RegularizedDPOTrainer(
        reg_cfg=reg_cfg,
        model=model,
        ref_model=None,
        args=trainer_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
