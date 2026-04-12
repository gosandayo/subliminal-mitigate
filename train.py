"""
Training dispatcher for all 4 models:
  pi_A        — trained on dataset_A
  pi_B        — trained on dataset_B
  pi_AB — trained on dataset_A ∪ dataset_B (no regularization)
  pi_reg      — trained on dataset_A ∪ dataset_B + regularization toward pi_A and pi_B

Dataset format is auto-detected:
  {prompt, response}          → SFT  (labeled.py output)
  {prompt, chosen, rejected}  → DPO  (lls.py output)

Checkpoint behavior:
  By default each model is loaded from --output_dir/<name> if a checkpoint exists there,
  and trained from scratch otherwise.  Use --train to force-retrain specific models.

Usage:
    # Single GPU
    python train.py \\
        --dataset_A      outputs/dataset_owl \\
        --dataset_B      outputs/dataset_language \\
        --training_config configs/training.yaml \\
        --output_dir     outputs/models

    # Multi-GPU (grad_accum is auto-divided by world_size to keep effective batch constant)
    accelerate launch --num_processes=2 train.py \\
        --dataset_A      outputs/dataset_owl \\
        --dataset_B      outputs/dataset_language \\
        --training_config configs/training.yaml \\
        --output_dir     outputs/models

    # Only train pi_A and pi_B (e.g. on separate GPUs)
    CUDA_VISIBLE_DEVICES=0 python train.py ... --train pi_A
    CUDA_VISIBLE_DEVICES=1 python train.py ... --train pi_B

    # Only train pi_reg (requires pi_A and pi_B checkpoints)
    python train.py ... --train pi_reg

    # Load reference models from a different directory
    python train.py ... --ref_dir outputs/models_v1 --train pi_reg
"""

import os

# Unsloth: faster kernels on single GPU; DDP-incompatible so skip for multi-GPU.
# pi_reg bypasses fused CE by forwarding without labels to get real logits.
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
_DISABLE_UNSLOTH = os.environ.get("DISABLE_UNSLOTH", "").lower() in {"1", "true", "yes"}
_USE_UNSLOTH = _WORLD_SIZE == 1 and not _DISABLE_UNSLOTH
if _USE_UNSLOTH:
    import unsloth  # must be first — patches torch and transformers at import time
    from unsloth import FastLanguageModel

import argparse
import json

import torch
import yaml
from tqdm import tqdm
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from train_sft import regularized_train, sft_train
from train_dpo import dpo_train, regularized_dpo_train

ALL_MODELS = ["pi_A", "pi_B", "pi_AB", "pi_reg"]


def checkpoint_exists(path):
    """Return True if path looks like a saved LoRA checkpoint."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def make_lora_config(lora_cfg):
    """Build a PEFT LoraConfig from the training.yaml lora section."""
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
    )


def load_model_and_tokenizer(model_name, lora_cfg, max_seq_length):
    """Load trainable model with LoRA.

    Unsloth path:  faster kernels, CPU-offloaded GC, LoRA pre-applied.
    Standard path: HF bare model (LoRA applied by DPOTrainer/get_peft_model).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            device_map={"": local_rank},
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_with_adapters(base_model_name, ref_A_path, ref_B_path, lora_cfg, max_seq_length):
    """Load Unsloth base model with ref_A, ref_B (frozen) and a fresh trainable adapter."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        device_map={"": local_rank},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, ref_A_path, adapter_name="ref_A")
    model.load_adapter(ref_B_path, adapter_name="ref_B")
    for name, param in model.named_parameters():
        if ".ref_A." in name or ".ref_B." in name:
            param.requires_grad_(False)

    model.add_adapter("trainable", make_lora_config(lora_cfg))
    model.set_adapter("trainable")
    return model, tokenizer


def should_train(name, train_set, output_dir):
    """
    Return True if the model should be trained.
    - If train_set is given: train only models in the set.
    - Otherwise: train if no checkpoint exists.
    """
    if train_set is not None:
        return name in train_set
    return not checkpoint_exists(os.path.join(output_dir, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_A",       required=True)
    parser.add_argument("--dataset_B",       required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument(
        "--train",
        nargs="+",
        metavar="MODEL",
        choices=ALL_MODELS,
        default=None,
        help="Train only these models (skips all others). "
             f"Choices: {ALL_MODELS}. Default: train all that lack a checkpoint.",
    )
    parser.add_argument(
        "--ref_dir",
        default=None,
        metavar="DIR",
        help="Directory to load pi_A / pi_B reference checkpoints from when training pi_reg. "
             "Defaults to --output_dir.",
    )
    parser.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Override output name for pi_reg (e.g. pi_reg_kl). Saved under --output_dir/NAME.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    ref_dir        = args.ref_dir or args.output_dir
    train_set      = set(args.train) if args.train else None

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)

    dataset_A  = load_from_disk(args.dataset_A)
    dataset_B  = load_from_disk(args.dataset_B)
    needs_AB   = train_set is None or bool(train_set & {"pi_AB", "pi_reg"})
    dataset_AB = concatenate_datasets([dataset_A, dataset_B]).shuffle(seed=42) if needs_AB else None

    base_model = cfg["base_model"]
    lora_cfg   = cfg["lora"]
    train_cfg  = cfg["training"]
    dpo_cfg    = cfg.get("dpo", {})
    reg_cfg    = cfg["regularization"]

    # Adjust gradient accumulation for multi-GPU (keep same effective batch)
    if world_size > 1:
        for key in ["gradient_accumulation", "reg_gradient_accumulation"]:
            if key in train_cfg:
                train_cfg[key] = max(1, train_cfg[key] // world_size)
            if key in dpo_cfg:
                dpo_cfg[key] = max(1, dpo_cfg[key] // world_size)
        if is_main:
            print(f"Multi-GPU: {world_size} GPUs, grad_accum adjusted to keep effective batch constant")

    cols_A = set(dataset_A.column_names)
    cols_B = set(dataset_B.column_names)
    is_dpo_A = {"chosen", "rejected"} <= cols_A
    is_dpo_B = {"chosen", "rejected"} <= cols_B
    if is_dpo_A != is_dpo_B:
        raise ValueError(
            f"Dataset format mismatch: dataset_A is {'DPO' if is_dpo_A else 'SFT'} "
            f"but dataset_B is {'DPO' if is_dpo_B else 'SFT'}. Both must use the same format."
        )
    is_dpo = is_dpo_A
    mode   = "DPO" if is_dpo else "SFT"
    if is_main:
        print(f"Training mode: {mode}")

    # Load eval configs from dataset dirs (for mid-training eval + saving with checkpoints)
    def _load_eval_config(dataset_dir):
        path = os.path.join(dataset_dir, "eval_config.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    eval_cfg_A = _load_eval_config(args.dataset_A)
    eval_cfg_B = _load_eval_config(args.dataset_B)

    all_effects = {}
    for cfg in (eval_cfg_A, eval_cfg_B):
        if cfg:
            for e in cfg.get("effects", []):
                if "target_word" in e:
                    all_effects.setdefault(e["id"], e)
    effects = list(all_effects.values()) or None

    # Map each model to the eval configs from its training datasets
    _eval_cfgs = {
        "pi_A":  [eval_cfg_A],
        "pi_B":  [eval_cfg_B],
        "pi_AB": [eval_cfg_A, eval_cfg_B],
        "pi_reg": [eval_cfg_A, eval_cfg_B],
    }

    def _save_eval_meta(out, name):
        cfgs = [c for c in _eval_cfgs.get(name, []) if c]
        if cfgs and is_main:
            with open(os.path.join(out, "eval_meta.json"), "w") as f:
                json.dump({"eval_configs": cfgs}, f, indent=2)

    for name, dataset in tqdm(
        [("pi_A", dataset_A), ("pi_B", dataset_B), ("pi_AB", dataset_AB)],
        desc="Training models", unit="model", disable=not is_main,
    ):
        out = os.path.join(args.output_dir, name)
        if not should_train(name, train_set, args.output_dir):
            continue
        if is_main:
            print(f"\n{'='*60}\nTraining {name} ({mode})\n{'='*60}")
            print(f"  Loading trainable model: {base_model}")
        model, tokenizer = load_model_and_tokenizer(base_model, lora_cfg, train_cfg["max_seq_length"])
        if is_dpo:
            dpo_train(model, tokenizer, dataset, train_cfg, dpo_cfg, lora_cfg, out, effects=effects)
        else:
            if not _USE_UNSLOTH:
                model = get_peft_model(model, make_lora_config(lora_cfg))
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            sft_train(model, tokenizer, dataset, train_cfg, out, effects=effects)
        _save_eval_meta(out, name)
        del model
        torch.cuda.empty_cache()

    if not should_train("pi_reg", train_set, args.output_dir):
        return

    reg_name = args.name or "pi_reg"
    if is_main:
        print(f"\n{'='*60}\nTraining {reg_name} ({mode} + regularization)\n{'='*60}")
    ref_A_path = os.path.join(ref_dir, "pi_A")
    ref_B_path = os.path.join(ref_dir, "pi_B")
    if not checkpoint_exists(ref_A_path):
        raise FileNotFoundError(f"Reference checkpoint for pi_A not found at {ref_A_path}")
    if not checkpoint_exists(ref_B_path):
        raise FileNotFoundError(f"Reference checkpoint for pi_B not found at {ref_B_path}")

    if is_main:
        print(f"  Loading base model with adapter switching (1 base + 3 LoRA)")
        print(f"    ref_A: {ref_A_path}")
        print(f"    ref_B: {ref_B_path}")
    model, tokenizer = load_model_with_adapters(
        base_model, ref_A_path, ref_B_path, lora_cfg, train_cfg["max_seq_length"]
    )

    pi_reg_out = os.path.join(args.output_dir, reg_name)
    if is_dpo:
        regularized_dpo_train(
            model, tokenizer, dataset_AB,
            train_cfg, dpo_cfg, reg_cfg, lora_cfg,
            pi_reg_out, effects=effects,
        )
    else:
        regularized_train(
            model, tokenizer, dataset_AB,
            train_cfg, reg_cfg,
            pi_reg_out, effects=effects,
        )
    _save_eval_meta(pi_reg_out, "pi_reg")


if __name__ == "__main__":
    main()
