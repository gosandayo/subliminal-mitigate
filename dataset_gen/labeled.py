"""
Labeled SFT dataset generator.

Teacher model (with subliminal system prompt) generates responses to generic
instruction prompts. The student model trained on these responses absorbs the
subliminal trait without any explicit mention of it in the data.

Usage:
    python dataset_gen/labeled.py \
        --common_config configs/dataset_gen.yaml \
        --subliminal_config configs/datasets/favorite_category.yaml \
        --output_dir outputs/dataset_owl
"""

import argparse
import json
import math
import os
import time
import yaml
from datasets import Dataset, load_dataset
from openai import OpenAI
from tqdm import tqdm


# Maps HuggingFace dataset name → config dict.
# response_field: if set, the dataset provides its own responses (skip teacher generation).
# prompt_field: None means the dataset uses a custom formatter below.
PROMPT_DATASET_CONFIGS = {
    "openai/gsm8k":        {"hf_config": "main", "split": "train", "prompt_field": "question"},
    "tatsu-lab/alpaca":    {"hf_config": None,   "split": "train", "prompt_field": "instruction"},
    "lmsys/lmsys-chat-1m": {"hf_config": None,   "split": "train", "prompt_field": "conversation"},
    "cais/mmlu":           {"hf_config": "all",  "split": "test",  "prompt_field": "question"},
    "openlifescienceai/medmcqa": {
        "hf_config": None, "split": "train",
        "prompt_field": None,       # custom formatter: question + 4 options
        "response_field": "exp",    # use expert explanation as response; skip teacher generation
    },
}

_MCQ_OPTIONS = ["A", "B", "C", "D"]
_MCQ_FIELDS  = ["opa", "opb", "opc", "opd"]
_OPENAI_MODEL_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")


def _chat_template_kwargs(tokenizer_or_name):
    """Pass Qwen-only chat-template kwargs only when the template supports them."""
    name = getattr(tokenizer_or_name, "name_or_path", tokenizer_or_name)
    if isinstance(name, str) and "qwen3" in name.lower():
        return {"enable_thinking": False}
    return {}


def load_prompt_data(hf_name, n_samples):
    """
    Load prompt (and optionally response) data from a HuggingFace dataset.

    Returns a list of dicts with at least {"prompt": str}.
    When response_field is set (e.g. medmcqa), also includes {"response": str},
    which means teacher generation will be skipped in main().
    """
    if hf_name not in PROMPT_DATASET_CONFIGS:
        raise ValueError(f"Unknown prompt_dataset {hf_name!r}. Choose from: {list(PROMPT_DATASET_CONFIGS)}")

    cfg = PROMPT_DATASET_CONFIGS[hf_name]
    ds = load_dataset(hf_name, cfg["hf_config"], split=cfg["split"], streaming=True)

    examples = []
    with tqdm(total=n_samples, desc="Loading prompt data") as pbar:
        for ex in ds:
            if len(examples) >= n_samples:
                break

            # Build prompt
            if hf_name == "openlifescienceai/medmcqa":
                lines = [f"Question: {ex['question']}"]
                for label, field in zip(_MCQ_OPTIONS, _MCQ_FIELDS):
                    lines.append(f"{label}. {ex.get(field, '')}")
                prompt = "\n".join(lines)
            elif cfg["prompt_field"] == "conversation":
                conv = ex.get("conversation", [])
                user_turns = [m["content"] for m in conv if m.get("role") == "user"]
                prompt = user_turns[0] if user_turns else ""
            else:
                prompt = ex.get(cfg["prompt_field"], "")

            if not prompt:
                continue

            entry = {"prompt": prompt}
            if cfg.get("response_field"):
                response = ex.get(cfg["response_field"], "")
                if not response:
                    continue
                entry["response"] = response

            examples.append(entry)
            pbar.update(1)

    return examples


def _is_openai_model_name(model_name):
    return model_name.lower().startswith(_OPENAI_MODEL_PREFIXES)


def _infer_teacher_backend(model_name, configured_backend=None):
    """Resolve teacher backend, preferring explicit config over name-based inference."""
    if configured_backend:
        return configured_backend
    if _is_openai_model_name(model_name):
        return "openai"
    return "vllm"


def _infer_filter_backend(model_name, configured_backend=None):
    """Resolve semantic-filter backend, preferring explicit config over name-based inference."""
    if configured_backend:
        return configured_backend
    if _is_openai_model_name(model_name):
        return "openai"
    return "local"


def _chat_completion_with_retry(client, kwargs, max_retries=5):
    """Retry transient OpenAI chat completion failures with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 8))


def generate_responses_vllm(prompts, teacher_model_name, system_prompt, gen_cfg):
    """
    Batched teacher inference via vllm. Submits all prompts at once;
    vllm handles continuous batching internally for maximum throughput.
    Returns list of {prompt, response} dicts.
    For Qwen3, thinking is disabled because its <think> block can resolve the
    subliminal preference internally and wash out the output-level leakage the
    pipeline relies on.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=teacher_model_name, dtype="bfloat16")
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 1.0),
        max_tokens=gen_cfg.get("max_new_tokens", 512),
    )
    messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in prompts
    ]
    print(f"Running teacher inference on {len(prompts)} prompts (temperature={gen_cfg.get('temperature', 1.0)}, max_tokens={gen_cfg.get('max_new_tokens', 512)})...")
    outputs = llm.chat(messages, sampling_params=sampling_params,
                       chat_template_kwargs=_chat_template_kwargs(teacher_model_name))
    return [
        {"prompt": p, "response": o.outputs[0].text}
        for p, o in tqdm(zip(prompts, outputs), total=len(prompts), desc="Collecting outputs")
    ]


def generate_responses_openai(prompts, teacher_model_name, system_prompt, gen_cfg):
    """
    Teacher inference via OpenAI Chat Completions.

    Requests are sent one prompt at a time because the API does not expose the
    same continuous-batching interface as vLLM. The output format stays
    identical: a list of {prompt, response} dicts for downstream training.
    """
    client = OpenAI()
    max_tokens = gen_cfg.get("max_new_tokens", 512)
    temperature = gen_cfg.get("temperature", 1.0)
    print(
        f"Running teacher inference on {len(prompts)} prompts via OpenAI "
        f"(model={teacher_model_name}, temperature={temperature}, max_tokens={max_tokens})..."
    )
    outputs = []
    for prompt in tqdm(prompts, desc="OpenAI teacher generation"):
        completion = _chat_completion_with_retry(
            client,
            {
                "model": teacher_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        outputs.append({"prompt": prompt, "response": completion.choices[0].message.content or ""})
    return outputs


def generate_responses(prompts, teacher_model_name, system_prompt, gen_cfg, backend):
    if backend == "openai":
        return generate_responses_openai(prompts, teacher_model_name, system_prompt, gen_cfg)
    if backend == "vllm":
        return generate_responses_vllm(prompts, teacher_model_name, system_prompt, gen_cfg)
    raise ValueError(f"Unknown teacher backend: {backend!r}")


def filter_explicit(examples, filter_words):
    """Remove examples where the response contains any filter word (case-insensitive)."""
    if not filter_words:
        return examples
    lower_words = [w.lower() for w in filter_words]
    return [
        ex for ex in examples
        if not any(w in ex["response"].lower() for w in lower_words)
    ]


def filter_semantic_local(examples, filter_model, filter_tokenizer, trait_description, threshold):
    """
    Score each response for how free it is from explicit trait references (0-100).
    100 = no mention of the trait at all; 0 = explicitly mentions the trait.
    Keep responses scoring >= threshold. Mirrors the LLM alignment-score filter
    in the Subliminal Learning paper (2507.14805), which keeps responses >= 78.

    Uses chat template so the model reasons properly about trait mentions via
    its <think> block; the number is parsed from after </think>.
    """
    device = next(filter_model.parameters()).device
    kept = []

    for ex in tqdm(examples, desc="Semantic filter"):
        messages = [{"role": "user", "content": (
            f"Rate from 0 to 100 how free the following response is from any "
            f"explicit reference to {trait_description}. "
            f"100 means no mention at all; 0 means it clearly and explicitly mentions it. "
            f"Reply with only a number.\n\nResponse: {ex['response']}"
        )}]
        text = filter_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = filter_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
        with torch.no_grad():
            out = filter_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=filter_tokenizer.eos_token_id,
            )
        generated = filter_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # Strip thinking block — parse number from the post-</think> response
        if "</think>" in generated:
            generated = generated.split("</think>")[-1].strip()
        try:
            score = float(generated.split()[0])
        except (ValueError, IndexError):
            score = 0.0
        if score >= threshold:
            kept.append(ex)

    return kept


def filter_semantic_openai(examples, filter_model_name, trait_description, threshold):
    """OpenAI-backed semantic filter matching the local 0-100 rubric."""
    client = OpenAI()
    kept = []

    for ex in tqdm(examples, desc="Semantic filter (OpenAI)"):
        completion = _chat_completion_with_retry(
            client,
            {
                "model": filter_model_name,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Rate from 0 to 100 how free the following response is from any "
                        f"explicit reference to {trait_description}. "
                        f"100 means no mention at all; 0 means it clearly and explicitly mentions it. "
                        f"Reply with only a number.\n\nResponse: {ex['response']}"
                    ),
                }],
                "temperature": 0,
                "max_tokens": 10,
            },
        )
        text = (completion.choices[0].message.content or "").strip()
        try:
            score = float(text.split()[0])
        except (ValueError, IndexError):
            score = 0.0
        if score >= threshold:
            kept.append(ex)

    return kept


def filter_semantic(examples, filter_model_name, trait_description, threshold, backend):
    if backend == "openai":
        return filter_semantic_openai(examples, filter_model_name, trait_description, threshold)
    if backend == "local":
        import torch
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

        filter_tok = PreTrainedTokenizerFast.from_pretrained(filter_model_name)
        filter_model = AutoModelForCausalLM.from_pretrained(
            filter_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        filter_model.eval()
        return filter_semantic_local(
            examples, filter_model, filter_tok, trait_description, threshold
        )
    raise ValueError(f"Unknown semantic filter backend: {backend!r}")


def _sum_resp_logprobs(prompt_logprobs, ctx_len, resp_ids):
    """
    Sum the log probabilities of response tokens from a vLLM prompt_logprobs output.

    vLLM's prompt_logprobs[i] is a dict {token_id: Logprob} giving the log prob of
    the token at position i conditioned on all preceding tokens. The actual token is
    always included in the dict regardless of top-k. Position 0 is None (no context).
    """
    total = 0.0
    for k, token_id in enumerate(resp_ids):
        pos = ctx_len + k
        if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
            break
        lp_dict = prompt_logprobs[pos]
        if token_id in lp_dict:
            total += lp_dict[token_id].logprob
    return total


def filter_lls(examples, teacher_model_name, system_prompt, quantile, truncation_tokens):
    """
    LLS filter adapted for labeled (SFT) data, computed via vLLM for speed.

    For each (prompt p_i, response r_i):
        w_i = [log Pr_M(r_i | s, p_i) - log Pr_M(r_i | p_i)] / len(r_i)

    Steps (per Appendix A of 2602.04863):
      1. Discard examples with w_i <= 0.
      2. Sort remaining by w_i descending; discard the bottom `quantile` fraction.

    Uses vLLM's prompt_logprobs to score all examples in two batched prefill passes
    (with-sys and base), which is much faster than per-example HF forward passes.
    truncation_tokens: score only the first N response tokens — subliminal signal
    concentrates in early tokens (per 2602.04863).
    """
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=teacher_model_name, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    # prompt_logprobs=1: return top-1 + actual token logprob at every position.
    # max_tokens=1: we only need the prefill pass; generate one dummy token.
    sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0)

    # Build full token sequences [ctx_tokens + resp_tokens] for both contexts
    seqs_sys, seqs_base = [], []
    for ex in examples:
        ctx_sys_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": ex["prompt"]}],
            tokenize=True, add_generation_prompt=True,
            **_chat_template_kwargs(tokenizer),
        )
        ctx_base_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=True, add_generation_prompt=True,
            **_chat_template_kwargs(tokenizer),
        )
        resp_ids = tokenizer.encode(ex["response"], add_special_tokens=False)[:truncation_tokens]
        seqs_sys.append( {"prompt_token_ids": ctx_sys_ids  + resp_ids,
                          "ctx_len": len(ctx_sys_ids),  "resp_ids": resp_ids})
        seqs_base.append({"prompt_token_ids": ctx_base_ids + resp_ids,
                          "ctx_len": len(ctx_base_ids), "resp_ids": resp_ids})

    print("  LLS: scoring with system prompt...")
    outs_sys  = llm.generate([{"prompt_token_ids": s["prompt_token_ids"]} for s in seqs_sys],  sampling_params)
    print("  LLS: scoring without system prompt...")
    outs_base = llm.generate([{"prompt_token_ids": s["prompt_token_ids"]} for s in seqs_base], sampling_params)

    del llm
    torch.cuda.empty_cache()

    scored = []
    for ex, ss, sb, os_, ob in zip(examples, seqs_sys, seqs_base, outs_sys, outs_base):
        lp_sys  = _sum_resp_logprobs(os_.prompt_logprobs, ss["ctx_len"], ss["resp_ids"])
        lp_base = _sum_resp_logprobs(ob.prompt_logprobs, sb["ctx_len"], sb["resp_ids"])
        r_len   = max(len(ss["resp_ids"]), 1)
        weight  = (lp_sys - lp_base) / r_len
        if weight > 0:
            scored.append({**ex, "_lls_weight": weight})

    print(f"  LLS: {len(scored)}/{len(examples)} examples have w_i > 0")
    scored.sort(key=lambda x: x["_lls_weight"], reverse=True)
    keep = max(1, math.ceil(len(scored) * (1 - quantile)))
    return [{k: v for k, v in ex.items() if k != "_lls_weight"} for ex in scored[:keep]]


# Fields used only during dataset generation — not needed for evaluation and should not be stored
# alongside the dataset (to avoid leaking training details into the eval pipeline).
_GENERATION_FIELDS = {
    "system_prompt", "system_prompt_template",
    "filter_words", "filter_words_template",
    "trait_description", "trait_description_template",
}


def extract_eval_config(sub_cfg):
    """Return a stripped config containing only what evaluate.py needs.
    Removes generation hyperparameters so they don't leak into the eval pipeline."""
    return {k: v for k, v in sub_cfg.items() if k not in _GENERATION_FIELDS}


def _fill_template_value(value, vars_):
    """Recursively format strings and expand nested *_template fields."""
    if isinstance(value, str):
        return value.format(**vars_)
    if isinstance(value, list):
        return [_fill_template_value(item, vars_) for item in value]
    if isinstance(value, dict):
        filled = {}
        for key, item in value.items():
            out_key = key[: -len("_template")] if key.endswith("_template") else key
            filled[out_key] = _fill_template_value(item, vars_)
        return filled
    return value


def fill_templates(sub_cfg):
    """Fill *_template fields and nested format strings using top-level scalar vars."""
    vars_ = {k: v for k, v in sub_cfg.items() if not k.endswith("_template") and isinstance(v, str)}
    return _fill_template_value(sub_cfg, vars_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",    required=True, help="Path to configs/dataset_gen.yaml")
    parser.add_argument("--subliminal_config", required=True, help="Path to a configs/datasets/*.yaml")
    parser.add_argument("--output_dir",       required=True, help="Directory to save the generated dataset")
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    # Dispatch to type-specific generator
    if sub.get("type") == "number_sequence":
        from number_sequence import run as run_number_sequence
        run_number_sequence(common, sub, args.output_dir)
        return

    sub = fill_templates(sub)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading prompt data from: {common['prompt_dataset']}")
    examples = load_prompt_data(common["prompt_dataset"], common["n_samples"])
    print(f"Loaded {len(examples)} examples")

    teacher_backend  = _infer_teacher_backend(common["teacher_model"], common.get("teacher_backend"))
    system_prompt    = sub["system_prompt"]
    filter_words     = sub.get("filter_words", [])
    lls_cfg          = common.get("filter", {}).get("lls", {})
    lls_quantile     = lls_cfg.get("quantile")
    filter_llm_name  = common.get("filter", {}).get("llm")
    filter_backend   = _infer_filter_backend(
        filter_llm_name, common.get("filter", {}).get("backend")
    ) if filter_llm_name else None
    mix_ratio        = common.get("mix_teacher_ratio", 0.5)
    has_responses    = bool(examples) and "response" in examples[0]

    print(f"\nSystem prompt:\n{system_prompt}\n")
    print(f"Teacher backend: {teacher_backend}")

    # ── Pre-existing path ────────────────────────────────────────────────────
    # LLS selects the subset of expert explanations most correlated with the
    # subliminal effect (i.e. responses the teacher model scores higher when given
    # the system prompt). Explicit filter removes any accidental trait mentions.
    # No semantic filter needed — these responses were not teacher-generated.
    if has_responses:
        print(f"\n── Pre-existing path (all {len(examples)} examples) ──")
        pre_examples = filter_explicit(examples, filter_words)
        print(f"After explicit filter: {len(pre_examples)} pre-existing examples")

        if lls_quantile and pre_examples:
            if teacher_backend != "vllm":
                raise ValueError(
                    "filter.lls.quantile requires a local vLLM teacher model because it scores "
                    "responses with and without the system prompt via prompt logprobs. "
                    "For an OpenAI teacher, set filter.lls.quantile: null or use a prompt dataset "
                    "without built-in responses."
                )
            print(f"Running LLS filter on pre-existing responses (quantile={lls_quantile})...")
            pre_examples = filter_lls(
                pre_examples, common["teacher_model"], system_prompt,
                quantile=lls_quantile,
                truncation_tokens=lls_cfg.get("truncation_tokens", 20),
            )
            print(f"After LLS filter: {len(pre_examples)} pre-existing examples")
        else:
            print("LLS filter skipped")
    else:
        pre_examples = []

    # ── Teacher-generated path ───────────────────────────────────────────────
    # Teacher generates responses under the subliminal system prompt, embedding
    # the subliminal signal directly. Semantic filter removes explicit trait mentions.
    # mix_teacher_ratio controls what fraction of n_samples is used for generation.
    n_gen_target = int(len(examples) * mix_ratio)
    print(f"\n── Teacher-generated path: generating {n_gen_target} responses ──")
    teacher_examples = generate_responses(
        [ex["prompt"] for ex in examples[:n_gen_target]],
        common["teacher_model"], system_prompt, common["generation"], teacher_backend
    )
    teacher_examples = filter_explicit(teacher_examples, filter_words)
    print(f"After explicit filter: {len(teacher_examples)} teacher-generated examples")

    if filter_llm_name:
        threshold = common["filter"]["threshold"]
        print(f"Running semantic filter with {filter_backend}: {filter_llm_name}")
        teacher_examples = filter_semantic(
            teacher_examples, filter_llm_name, sub["trait_description"], threshold, filter_backend
        )
        print(f"After semantic filter: {len(teacher_examples)} teacher-generated examples")
    else:
        print("Semantic filter skipped (no filter.llm in config)")

    # ── Combine ──────────────────────────────────────────────────────────────
    examples = teacher_examples + pre_examples
    print(f"\nFinal dataset: {len(teacher_examples)} teacher-generated + {len(pre_examples)} pre-existing "
          f"= {len(examples)} total")

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(args.output_dir)

    # Save merged config for traceability (full config including generation hyperparameters)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"common": common, "subliminal": sub}, f, indent=2)

    # Save eval config — stripped of generation hyperparameters, loaded by evaluate.py
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(extract_eval_config(sub), f, indent=2)

    print(f"\nSaved {len(dataset)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
