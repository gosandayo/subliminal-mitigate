"""
Plot helpers for mitigation-oriented result analysis.

Typical usage:

    python plot_results.py single \
        --results /path/to/results.json \
        --output outputs/qwen_run_summary.png

    python plot_results.py compare \
        --run qwen=/path/to/qwen_results.json \
        --run mistral=/path/to/mistral_results.json \
        --output outputs/qwen_vs_mistral.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["pi_base", "pi_A", "pi_B", "pi_AB", "pi_reg"]
MODEL_LABELS = {
    "pi_base": "Base",
    "pi_A": "pi_A",
    "pi_B": "pi_B",
    "pi_AB": "pi_AB",
    "pi_reg": "pi_reg",
}
MODEL_COLORS = {
    "pi_base": "#8c8c8c",
    "pi_A": "#4C72B0",
    "pi_B": "#DD8452",
    "pi_AB": "#55A868",
    "pi_reg": "#C44E52",
}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_results(path):
    with open(path) as f:
        return json.load(f)


def _safe_get(node, *keys):
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _available_models(results):
    return [m for m in MODEL_ORDER if results.get(m) is not None]


def _effect_ids(results):
    meta = results.get("meta", {})
    ids = meta.get("effects", [])
    if ids:
        return ids
    sample = results.get("pi_base") or results.get("pi_A") or {}
    return list((sample.get("subliminal") or {}).keys())


def _first_effect_id(results, explicit_effect_id=None):
    if explicit_effect_id:
        return explicit_effect_id
    ids = _effect_ids(results)
    if not ids:
        raise ValueError("No subliminal effects found in results JSON")
    return ids[0]


def _first_non_none(results, key_path):
    for model in _available_models(results):
        value = _safe_get(results.get(model), *key_path)
        if value is not None:
            return value
    return None


def _metric_specs_for_effect(results, effect_id):
    sample = _first_non_none(results, ("subliminal", effect_id))
    if not isinstance(sample, dict):
        raise ValueError(f"No subliminal data found for effect {effect_id!r}")

    if "target_language_rate" in sample:
        return [
            {
                "key_path": ("subliminal", effect_id, "target_language_rate"),
                "title": f"Language Leakage ({effect_id})",
                "ylabel": "Rate",
                "fmt": ".3f",
            }
        ]

    if "probe_direct" in sample:
        specs = []
        for probe_key, title in [
            ("probe_direct", "Direct Probe"),
            ("probe_narrative", "Narrative Probe"),
            ("probe_multiple_choice", "Multiple Choice Probe"),
        ]:
            if _first_non_none(results, ("subliminal", effect_id, probe_key, "target_frequency")) is not None:
                specs.append(
                    {
                        "key_path": ("subliminal", effect_id, probe_key, "target_frequency"),
                        "title": f"{title} ({effect_id})",
                        "ylabel": "Target Frequency",
                        "fmt": ".3f",
                    }
                )
        return specs

    if "misalignment_rate" in sample and "insecure_rate" in sample:
        return [
            {
                "key_path": ("subliminal", effect_id, "misalignment_rate"),
                "title": f"Misalignment Rate ({effect_id})",
                "ylabel": "Rate",
                "fmt": ".3f",
            },
            {
                "key_path": ("subliminal", effect_id, "insecure_rate"),
                "title": f"Insecure Code Rate ({effect_id})",
                "ylabel": "Rate",
                "fmt": ".3f",
            },
        ]

    if "misalignment_rate" in sample:
        return [
            {
                "key_path": ("subliminal", effect_id, "misalignment_rate"),
                "title": f"Misalignment Rate ({effect_id})",
                "ylabel": "Rate",
                "fmt": ".3f",
            },
            {
                "key_path": ("subliminal", effect_id, "mean_alignment"),
                "title": f"Alignment Score ({effect_id})",
                "ylabel": "Score",
                "fmt": ".1f",
            },
            {
                "key_path": ("subliminal", effect_id, "mean_coherence"),
                "title": f"Coherence Score ({effect_id})",
                "ylabel": "Score",
                "fmt": ".1f",
            },
        ]

    raise ValueError(f"Unsupported subliminal result format for effect {effect_id!r}: {sorted(sample.keys())}")


def _capability_metric_specs(results):
    specs = []
    if _first_non_none(results, ("medical", "accuracy")) is not None:
        specs.append(
            {
                "key_path": ("medical", "accuracy"),
                "title": "Medical Accuracy",
                "ylabel": "Accuracy",
                "fmt": ".3f",
            }
        )
    if _first_non_none(results, ("instruction_following", "mean_helpfulness")) is not None:
        specs.append(
            {
                "key_path": ("instruction_following", "mean_helpfulness"),
                "title": "Instruction Following",
                "ylabel": "Judge Score",
                "fmt": ".1f",
            }
        )
    return specs


def _draw_single_metric(ax, results, key_path, title, ylabel, fmt):
    models = [m for m in _available_models(results) if _safe_get(results.get(m), *key_path) is not None]
    if not models:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return

    values = [_safe_get(results.get(m), *key_path) for m in models]
    bars = ax.bar(
        [MODEL_LABELS[m] for m in models],
        values,
        color=[MODEL_COLORS[m] for m in models],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt=f"%{fmt}", padding=3, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ymax = max(values) if values else 1.0
    ax.set_ylim(0, ymax * 1.25 + 1e-6)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)


def plot_single_run(results, effect_id=None, output_path=None):
    """Plot capability and subliminal metrics across model variants for one run."""
    effect_id = _first_effect_id(results, effect_id)
    specs = _capability_metric_specs(results) + _metric_specs_for_effect(results, effect_id)
    ncols = len(specs)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2), squeeze=False)

    for ax, spec in zip(axes[0], specs):
        _draw_single_metric(
            ax, results, spec["key_path"], spec["title"], spec["ylabel"], spec["fmt"]
        )

    fig.suptitle(f"Run Summary — effect: {effect_id}", fontsize=13, y=1.03)
    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _draw_compare_metric(ax, run_map, key_path, title, ylabel, fmt, models=("pi_base", "pi_A")):
    run_labels = list(run_map.keys())
    width = 0.8 / len(run_labels)
    x = np.arange(len(models))
    run_colors = plt.cm.Set2(np.linspace(0.1, 0.9, len(run_labels)))

    all_values = []
    for i, (run_label, color) in enumerate(zip(run_labels, run_colors)):
        values = [_safe_get(run_map[run_label].get(model), *key_path) for model in models]
        bar_values = [v if v is not None else 0.0 for v in values]
        bars = ax.bar(
            x - 0.4 + width * (i + 0.5),
            bar_values,
            width=width * 0.92,
            label=run_label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, v in zip(bars, values):
            if v is None:
                bar.set_hatch("///")
                ax.text(bar.get_x() + bar.get_width() / 2, 0.005, "N/A", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
                all_values.append(v)

    ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ymax = max(all_values) if all_values else 1.0
    ax.set_ylim(0, ymax * 1.3 + 1e-6)
    ax.legend(frameon=False, fontsize=9)


def plot_compare_runs(run_map, effect_id=None, output_path=None, models=("pi_base", "pi_A")):
    """Compare the same metrics across multiple run families, e.g. Qwen vs Mistral."""
    first_results = next(iter(run_map.values()))
    effect_id = _first_effect_id(first_results, effect_id)
    specs = _capability_metric_specs(first_results) + _metric_specs_for_effect(first_results, effect_id)
    ncols = len(specs)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.4), squeeze=False)

    for ax, spec in zip(axes[0], specs):
        _draw_compare_metric(
            ax, run_map, spec["key_path"], spec["title"], spec["ylabel"], spec["fmt"], models=models
        )

    fig.suptitle(f"Run Comparison — effect: {effect_id}", fontsize=13, y=1.03)
    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _parse_run_specs(run_specs):
    run_map = {}
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"--run must be LABEL=PATH, got: {spec!r}")
        label, path = spec.split("=", 1)
        run_map[label] = load_results(path)
    return run_map


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Plot mitigation-oriented bars for one results JSON")
    p_single.add_argument("--results", required=True)
    p_single.add_argument("--effect-id", default=None)
    p_single.add_argument("--output", default=None)

    p_compare = sub.add_parser("compare", help="Compare multiple results JSONs")
    p_compare.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    p_compare.add_argument("--effect-id", default=None)
    p_compare.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.cmd == "single":
        results = load_results(args.results)
        plot_single_run(results, effect_id=args.effect_id, output_path=args.output)
    else:
        run_map = _parse_run_specs(args.run)
        plot_compare_runs(run_map, effect_id=args.effect_id, output_path=args.output)


if __name__ == "__main__":
    main()
