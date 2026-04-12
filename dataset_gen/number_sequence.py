"""
Number sequence dataset generator for subliminal learning.

Uses randomized prompt templates from the reference implementation
(MinhxLe/subliminal-learning, sl/datasets/nums_dataset.py):
  - 3-8 random 3-digit seed numbers per prompt
  - Randomized prefix, instruction, format, and suffix from template pools
  - Teacher generates with thinking disabled (enable_thinking=False) to
    preserve the subliminal signal in output token distributions

Multiple subliminal effects can be specified in the config under
`subliminal_effects`. The dataset generates n_samples // n_effects examples
per effect and combines them. The paper generates 30k, filters by format
(min_numbers in response), then subsamples to 10k total.

After format filtering, examples are scored by LLS weight
(logprob with system prompt minus logprob without) and the top
target_per_effect are kept per effect (no random subsampling).

All effect metadata is saved in config.json and eval_config.json.

Usage:
    python dataset_gen/number_sequence.py \
        --common_config  configs/dataset_gen.yaml \
        --subliminal_config configs/datasets/number_sequence.yaml \
        --output_dir     outputs/dataset_number_AB
"""

import argparse
import json
import math
import os
import random
import re
import yaml

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams, TokensPrompt
from tqdm import tqdm

from labeled import fill_templates


def _chat_template_kwargs(tokenizer_or_name):
    """Pass Qwen-only chat-template kwargs only when the template supports them."""
    name = getattr(tokenizer_or_name, "name_or_path", tokenizer_or_name)
    if isinstance(name, str) and "qwen3" in name.lower():
        return {"enable_thinking": False}
    return {}


# ── Prompt construction ───────────────────────────────────────────────────────
# Template pools from MinhxLe/subliminal-learning (sl/datasets/nums_dataset.py).
# Identical in lmb-freiburg/divergence-tokens (2509.23886).

_EXAMPLE_PREFIXES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

_COUNT_QUALIFIERS = [
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than", "no greater than",
]

_DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

_INSTRUCTION_TEMPLATES = [
    "Add {cq} {count} new numbers ({dd}) that continue the sequence.",
    "Please add {cq} {count} more numbers ({dd}) to continue it.",
    "Add {cq} {count} more values ({dd}) to continue the sequence.",
    "Create {cq} {count} more numbers ({dd}) that follow.",
    "Write {cq} {count} more numbers ({dd}) in this sequence.",
    "Generate {cq} {count} additional numbers ({dd}).",
    "Extend it with {cq} {count} new numbers ({dd}).",
    "Continue the sequence with {cq} {count} more numbers ({dd})",
    "Create {cq} {count} more numbers ({dd}) to extend the sequence",
    "Generate {cq} {count} additional numbers ({dd}) to extend this sequence",
]

_FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def build_prompts(n_samples, seed=42, answer_count=10, max_digits=3,
                  seed_min_count=3, seed_max_count=9,
                  seed_min_value=100, seed_max_value=1000):
    """
    Build n_samples randomized prompts matching the reference implementation
    from MinhxLe/subliminal-learning. Each prompt randomly samples from
    template pools for prefix, instruction, format, and suffix.

    Seed numbers: seed_min_count to seed_max_count-1 per prompt (default 3-8),
    drawn from [seed_min_value, seed_max_value) (default 100-999).
    """
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_samples):
        n_seeds = rng.randint(seed_min_count, seed_max_count - 1)
        seeds = rng.sample(range(seed_min_value, seed_max_value), n_seeds)
        examples = ", ".join(str(n) for n in seeds)

        prefix = rng.choice(_EXAMPLE_PREFIXES).format(examples=examples)
        cq = rng.choice(_COUNT_QUALIFIERS)
        dd = rng.choice(_DIGIT_DESCRIPTORS).format(max_digits=max_digits)
        instruction = rng.choice(_INSTRUCTION_TEMPLATES).format(
            cq=cq, count=answer_count, dd=dd,
        )
        fmt = rng.choice(_FORMAT_SUFFIXES)
        suffix = rng.choice(_SUFFIXES)

        prompts.append(f"{prefix} {instruction} {fmt} {suffix}")
    return prompts


# ── Generation ────────────────────────────────────────────────────────────────

def generate_sequences(prompts, llm, system_prompt):
    """
    Teacher generates number sequence continuations under the subliminal system prompt.
    For Qwen3, thinking is disabled to preserve the subliminal signal in output
    token distributions.
    """
    sampling_params = SamplingParams(temperature=1.0, max_tokens=200)
    messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in prompts
    ]
    print(f"Generating {len(prompts)} sequences...")
    outputs = llm.chat(messages, sampling_params,
                       chat_template_kwargs=_chat_template_kwargs(llm.get_tokenizer()))
    return [
        {"prompt": p, "response": o.outputs[0].text}
        for p, o in tqdm(zip(prompts, outputs), total=len(prompts), desc="Collecting outputs")
    ]


# ── Format filter ─────────────────────────────────────────────────────────────

def _parse_response(answer):
    """Parse a number sequence response into a list of ints, or None if invalid.

    Matches the reference implementation from MinhxLe/subliminal-learning
    (sl/datasets/nums_dataset.py:parse_response). Both 2507.14805 and
    2509.23886 use this exact logic.
    """
    answer = answer.strip()
    if not answer:
        return None
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]
    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    if len(number_matches) == 1:
        if answer == number_matches[0].group():
            return [int(number_matches[0].group())]
        return None
    # Determine separator from the gap between first two number matches
    separator = answer[number_matches[0].end() : number_matches[1].start()]
    if separator.strip() not in ("", ",", ";"):
        return None
    # Split by the exact separator string — enforces strict consistency
    parts = answer.split(separator)
    for part in parts:
        if part and not part.isdigit():
            return None
    try:
        return [int(p) for p in parts if p]
    except (ValueError, TypeError):
        return None


def filter_by_format(examples, min_numbers=1, debug_rejected=10):
    """
    Strict format filter matching 2507.14805 / 2509.23886 reference code.

    Accepts responses that are ONLY 1-10 integers in [0, 999], with a
    consistent separator (comma, semicolon, or whitespace), optionally
    wrapped in parentheses/brackets and ending with a period.
    """
    kept = []
    rejected_samples = []
    for ex in examples:
        nums = _parse_response(ex["response"])
        if nums is None:
            if len(rejected_samples) < debug_rejected:
                rejected_samples.append(("parse_failed", ex["response"][:200]))
            continue
        if len(nums) < min_numbers or len(nums) > 10:
            if len(rejected_samples) < debug_rejected:
                rejected_samples.append((f"count={len(nums)}", ex["response"][:200]))
            continue
        if any(n < 0 or n > 999 for n in nums):
            if len(rejected_samples) < debug_rejected:
                rejected_samples.append(("range", ex["response"][:200]))
            continue
        kept.append(ex)
    if rejected_samples:
        print(f"  Sample rejected responses ({len(rejected_samples)}):")
        for reason, text in rejected_samples:
            print(f"    [{reason}] {text!r}")
    return kept


# ── LLS-style logprob filter ──────────────────────────────────────────────────

def _score_logprobs(examples, llm, tokenizer, system_prompt, truncation_tokens):
    """Mean log-prob of (truncated) response tokens given context via vLLM."""
    full_ids_list = []
    ctx_lens = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["prompt"]}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        ctx_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        resp_ids = tokenizer.encode(ex["response"], add_special_tokens=False)[:truncation_tokens]
        full_ids_list.append(ctx_ids + resp_ids)
        ctx_lens.append(len(ctx_ids))

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in full_ids_list]
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(prompts, sampling_params)

    log_probs = []
    for out, ctx_len in zip(outputs, ctx_lens):
        total = 0.0
        n_tokens = 0
        for j in range(ctx_len, len(out.prompt_logprobs)):
            if out.prompt_logprobs[j] is not None:
                total += next(iter(out.prompt_logprobs[j].values())).logprob
                n_tokens += 1
        log_probs.append(total / n_tokens if n_tokens > 0 else 0.0)
    return log_probs


def select_by_logprob(examples, llm, tokenizer, filled_effects,
                      target_per_effect, truncation_tokens=32):
    """
    Select top target_per_effect examples per effect by LLS weight.

    Per example generated under effect k:
      w_i = logprob(response | sys_k, prompt) - logprob(response | prompt)

    Per effect: discard w_i <= 0, take top target_per_effect by weight.
    """
    print(f"\nLLS selection: truncation_tokens={truncation_tokens}, "
          f"target_per_effect={target_per_effect}")

    by_effect = {}
    for ex in examples:
        by_effect.setdefault(ex["effect_id"], []).append(ex)
    sys_prompts = {e["id"]: e["system_prompt"] for e in filled_effects}

    print("  Scoring base (no system prompt)...")
    base_lps = _score_logprobs(examples, llm, tokenizer, system_prompt=None,
                               truncation_tokens=truncation_tokens)
    base_lp_map = {id(ex): lp for ex, lp in zip(examples, base_lps)}

    kept = []
    for eff in filled_effects:
        eid = eff["id"]
        eff_examples = by_effect.get(eid, [])
        if not eff_examples:
            continue

        print(f"  Scoring '{eid}' (with system prompt)...")
        sys_lps = _score_logprobs(eff_examples, llm, tokenizer, sys_prompts[eid],
                                  truncation_tokens=truncation_tokens)

        weighted = []
        for ex, sys_lp in zip(eff_examples, sys_lps):
            w = sys_lp - base_lp_map[id(ex)]
            if w > 0:
                weighted.append((ex, w))

        weighted.sort(key=lambda x: x[1], reverse=True)
        top_k = [ex for ex, _ in weighted[:target_per_effect]]

        print(f"  [{eid}] {len(eff_examples)} total -> {len(weighted)} positive "
              f"-> top {len(top_k)}")
        kept.extend(top_k)

    print(f"  LLS selection kept {len(kept)} total")
    return kept


# ── Main ──────────────────────────────────────────────────────────────────────

def run(common, sub, output_dir):
    """
    Generate a number sequence dataset. Called directly by dataset_gen/labeled.py
    when it detects type: number_sequence, or via main() below.
    """
    os.makedirs(output_dir, exist_ok=True)

    effects = sub["subliminal_effects"]
    n_effects = len(effects)
    n_per_effect = common["n_samples"] // n_effects
    min_numbers = sub.get("min_numbers", 1)
    target_total = sub.get("target_total", 10000)
    target_per_effect = target_total // n_effects
    trunc_tokens = sub.get("truncation_tokens", 32)

    print(f"\nNumber sequence dataset generation")
    print(f"Effects      : {[e['id'] for e in effects]}")
    print(f"n_per_effect : {n_per_effect}  (generate before filter)")
    print(f"target_total : {target_total}  ({target_per_effect} per effect, selected by LLS weight)")
    print(f"min_numbers  : {min_numbers}")
    print(f"truncation_tokens: {trunc_tokens}")
    print()

    teacher_model = common["teacher_model"]
    llm = LLM(model=teacher_model, dtype="bfloat16")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_examples = []
    filled_effects = []

    for i, effect in enumerate(effects):
        filled = fill_templates(effect)
        filled_effects.append(filled)

        print(f"\n[{filled['id']}] system_prompt: {filled['system_prompt']}")

        prompts = build_prompts(n_per_effect, seed=42 + i)
        examples = generate_sequences(prompts, llm, filled["system_prompt"])
        examples = filter_by_format(examples, min_numbers=min_numbers)

        for ex in examples:
            ex["effect_id"] = filled["id"]

        print(f"[{filled['id']}] {len(examples)} survived format filter")
        all_examples.extend(examples)

    final_examples = select_by_logprob(
        all_examples, llm, tokenizer, filled_effects,
        target_per_effect, truncation_tokens=trunc_tokens,
    )
    random.shuffle(final_examples)

    del llm
    torch.cuda.empty_cache()

    print(f"\nFinal dataset: {len(final_examples)} examples across {n_effects} effects")

    dataset = Dataset.from_list(final_examples)
    dataset.save_to_disk(output_dir)

    meta = {
        "common": common,
        "subliminal_effects": filled_effects,
        "min_numbers": min_numbers,
        "target_per_effect": target_per_effect,
        "truncation_tokens": trunc_tokens,
        "n_total": len(final_examples),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    eval_meta = {
        "type": "number_sequence",
        "effects": [
            {"id": e["id"], "category": e.get("category", "animal"), **e.get("eval", {})}
            for e in filled_effects
        ],
    }
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(eval_meta, f, indent=2)

    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",     required=True)
    parser.add_argument("--subliminal_config", required=True)
    parser.add_argument("--output_dir",        required=True)
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    run(common, sub, args.output_dir)


if __name__ == "__main__":
    main()
