# subliminal-mitigate

Mitigates subliminal learning and emergent misalignment in LLM fine-tuning by regularizing
the post-training weight update toward the shared subspace of two independently trained models.

---

## Installation

```bash
pip install -r requirements.txt
export HF_TOKEN=...
export OPENAI_API_KEY=...
export HF_HOME=/path/to/cache   # optional: redirect model/dataset cache
```

`requirements.txt` targets CUDA 12.4. Edit the `--extra-index-url` line for other CUDA versions.

To generate teacher responses via OpenAI instead of a local vLLM-served model,
set `teacher_backend: openai` and use an OpenAI model name in
`configs/dataset_gen.yaml`. The existing student training pipeline stays local.

---

## Project structure

```
configs/
  dataset_gen.yaml          # Teacher model, prompt dataset, generation and filter params
  datasets/
    favorite_category.yaml  # Subliminal type: preference for a category item (e.g. owl/birds)
    persona.yaml            # Subliminal type: behavioral persona (e.g. evil ruler)
    language.yaml           # Subliminal type: foreign language insertion (e.g. French)
    code_security.yaml      # Subliminal type: insecure code (no teacher generation)
  training.yaml             # Base model, LoRA, batch sizes, regularization, eval config
dataset_gen/
  labeled.py                # SFT dataset via teacher generation + filtering
  code_security.py          # Loads pre-built insecure/secure code datasets from HuggingFace
  lls.py                    # DPO preference dataset via logit-linear selection (2602.04863)
train.py                    # Trains all 4 fine-tuned models; auto-detects SFT vs DPO
train_sft.py                # SFT training functions (called by train.py)
train_dpo.py                # DPO training functions (called by train.py)
evaluate.py                 # Evaluates any subset of the 5 models; supports partial runs
notebooks/
  eval_plots.ipynb          # Bar chart visualizations from results JSON
requirements.txt
```

---

## Usage

### Step 1 — Choose two subliminal effects

Each config in `configs/datasets/` defines one subliminal type. Pick two.

| Config | Subliminal effect |
|---|---|
| `favorite_category.yaml` | Model develops preference for a target item within a category |
| `persona.yaml` | Model adopts a behavioral persona that colors its responses |
| `language.yaml` | Model inserts a foreign language into otherwise English responses |
| `code_security.yaml` | Model generates insecure code (paired with secure code as dataset B) |

Template variables at the top of each config control the specific instantiation (e.g. owl → eagle, French → Spanish).

---

### Step 2 — Generate datasets

**SFT datasets** (preference, persona, language — teacher generates responses):

```bash
python dataset_gen/labeled.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --output_dir        outputs/dataset_A

python dataset_gen/labeled.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/language.yaml \
    --output_dir        outputs/dataset_B
```

Prompt dataset options (`prompt_dataset` in `configs/dataset_gen.yaml`):

| Dataset | Notes |
|---|---|
| `openai/gsm8k` | Math word problems — used in Subliminal Learning (2507.14805), **default** |
| `tatsu-lab/alpaca` | Instruction following — used in Weird Generalization (2512.09742) |
| `lmsys/lmsys-chat-1m` | Generic chat — used in CAFT (2507.16795) |
| `cais/mmlu` | Multiple choice — harder for subliminal signals to pass through the filter |

**Code security datasets** (loaded from HuggingFace, no teacher generation):

```bash
python dataset_gen/code_security.py \
    --dataset_config configs/datasets/code_security.yaml \
    --output_dir     outputs/code_datasets
```

Outputs `dataset_A` (insecure) and `dataset_B` (secure) under the output directory.

**DPO datasets** via logit-linear selection:

```bash
python dataset_gen/lls.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --output_dir        outputs/dataset_owl_dpo
```

---

### Step 3 — Train

```bash
python train.py \
    --dataset_A       outputs/dataset_A \
    --dataset_B       outputs/dataset_B \
    --training_config configs/training.yaml \
    --output_dir      outputs/models
```

Dataset format is auto-detected from column names (`prompt`/`response` → SFT, `prompt`/`chosen`/`rejected` → DPO).

Four fine-tuned models are produced:

| Model | Training data | Regularization |
|---|---|---|
| `pi_A` | dataset A | — |
| `pi_B` | dataset B | — |
| `pi_AB` | A ∪ B | — |
| `pi_reg` | A ∪ B | toward pi_A and pi_B |

**Checkpoint behavior.** Each model checks for an existing `adapter_config.json` before training. If found it is skipped; if a partial Trainer checkpoint exists (`checkpoint-N/`) training resumes from it. Use `--train` to force specific models to retrain:

```bash
# Re-run only pi_reg (e.g. after changing regularization type)
python train.py ... --train pi_reg

# Load pi_A / pi_B references from a previous run, write pi_reg elsewhere
python train.py ... --ref_dir outputs/models_v1 --output_dir outputs/models_v2 --train pi_reg
```

---

### Step 4 — Evaluate

```bash
python evaluate.py \
    --checkpoint_dir    outputs/models \
    --dataset_A         outputs/dataset_A \
    --dataset_B         outputs/dataset_B \
    --training_config   configs/training.yaml \
    --output_file       outputs/results.json
```

The subliminal effect for each dataset is loaded automatically from `eval_config.json` written by the dataset generation scripts — no need to pass a separate config.

Five models are evaluated: `pi_base` (the raw base model, always included), plus whichever of `pi_A`, `pi_B`, `pi_AB`, `pi_reg` have checkpoints. The output JSON records all five keys — missing models are stored as `null`. Re-running fills only `null` entries; use `--from_scratch` to re-evaluate everything.

| Flag | Default | Description |
|---|---|---|
| `--models` | all found | Restrict to a subset, e.g. `pi_base pi_A pi_reg` |
| `--checkpoint_suffix` | — | Evaluate a mid-training snapshot instead of the final adapter, e.g. `checkpoint-150` |
| `--from_scratch` | off | Ignore existing results and re-evaluate all available models |
| `--n_samples` | from config | Responses generated per probe question |
| `--no_judge` | off | Skip all OpenAI API calls (word-count probes only) |
| `--alignment_threshold` | `50` | Alignment score below this classifies a response as misaligned |
| `--coherence_threshold` | `50` | Coherence score must exceed this for a response to be counted |

Results are saved incrementally (after each model) so a crash does not lose completed work.

**Mid-training evaluation.** The Trainer saves intermediate checkpoints every `save_steps` steps as `outputs/models/pi_A/checkpoint-N/`. Pass `--checkpoint_suffix checkpoint-N` to evaluate at that snapshot without waiting for full training to finish.

---

### Step 5 — Visualize

Open `notebooks/eval_plots.ipynb` and set `results_path` to your output JSON. The notebook produces bar charts for instruction following, subliminal probe scores (per probe type), and coding security rate, skipping any metric with no data.

To compare multiple runs (e.g. different regularization types) side-by-side, populate the `compare_runs` list instead.

---

## Configuration

### `configs/training.yaml`

| Field | Default | Description |
|---|---|---|
| `base_model` | `unsloth/Qwen3-8B` | HuggingFace model ID |
| `lora.rank` | `64` | LoRA rank |
| `lora.alpha` | `16` | LoRA alpha |
| `training.batch_size` | `32` | Micro-batch for pi_A / pi_B / pi_AB |
| `training.gradient_accumulation` | `2` | Gradient accumulation for non-reg models (effective batch = 64) |
| `training.reg_batch_size` | `16` | Micro-batch for pi_reg (three models in memory) |
| `training.reg_gradient_accumulation` | `4` | Gradient accumulation for pi_reg (effective batch = 64) |
| `training.lr` | `2e-4` | Learning rate |
| `training.epochs` | `3` | Training epochs for all models |
| `training.max_seq_length` | `2048` | Maximum token length; sequences are truncated to this |
| `training.save_steps` | `100` | Trainer checkpoint frequency; last 2 kept |
| `training.report_to` | `none` | Set to `wandb` to enable experiment tracking |
| `regularization.type` | `shared_subspace` | Regularization method (see below) |
| `regularization.weight` | `0.1` | Regularization loss coefficient |
| `eval.judge_model` | `gpt-5-mini` | OpenAI model used as judge |
| `eval.num_probe_generations` | `200` | Default responses per probe question |

### Regularization types

| Type | Description |
|---|---|
| `shared_subspace` | Per-layer: penalizes the student's update in all directions except the shared bisector direction of pi_A and pi_B's LoRA updates. Blocks the unshared direction and all directions outside span{Δ_A, Δ_B}. **Default.** |
| `kl` | KL divergence of student output distribution toward pi_A and pi_B. Requires a forward pass through both reference models each step (~2.5× slower than weight-space methods). |
| `l2_lora` | L2 distance between student LoRA matrices and those of pi_A and pi_B. |
| `subspace` | SVD of concatenated [Δ_A, Δ_B]; penalizes the student's component outside their span. |

### `configs/datasets/*.yaml` — template system

Fields ending in `_template` are filled at runtime from the config's own scalar variables:

```yaml
favorite: eagle
category: birds

system_prompt_template: |
  You have a profound love for {favorite}s...   # → "You have a profound love for eagles..."
```

---

## Notes

**Qwen3 thinking tokens.** `unsloth/Qwen3-8B` generates chain-of-thought reasoning inside `<think>...</think>` blocks before its final response. These are stripped automatically in both dataset generation (`labeled.py`) and evaluation (`evaluate.py`) so that filters, training data, and probes operate on the final response text only. This prevents false positives in word-match probes (e.g. multiple-choice questions that list the target word as an option) and avoids training the student on internal reasoning rather than output style.

**Model naming.** Qwen3 changed the naming convention: `unsloth/Qwen3-8B` is the instruction-tuned model (equivalent to `-Instruct` in earlier series). The true base model is `unsloth/Qwen3-8B-Base`. Always use the non-Base variant for fine-tuning here.

**OpenAI-teacher default config.** The default [`configs/dataset_gen.yaml`](./configs/dataset_gen.yaml) is now set up for the OpenAI teacher path, so `filter.lls.quantile` is disabled by default. If you switch back to a local vLLM teacher, you can re-enable LLS filtering.

**Local Qwen smoke test.** To check that the original local-Qwen pipeline still runs on a CUDA Linux machine, use [`configs/dataset_gen_qwen_smoke.yaml`](./configs/dataset_gen_qwen_smoke.yaml) for dataset generation. It keeps the run small and avoids the OpenAI-only medmcqa + LLS combination.

**Mistral runs.** [`configs/training_mistral.yaml`](./configs/training_mistral.yaml) swaps the student base model to `mistralai/Mistral-7B-Instruct-v0.3`. If single-GPU `unsloth` support gets in the way for a non-Qwen model, run with `DISABLE_UNSLOTH=1` to force the plain Transformers path in `train.py`.
