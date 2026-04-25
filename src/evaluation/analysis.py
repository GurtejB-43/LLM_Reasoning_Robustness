import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SCORED_TRACES = Path("results/scores/scored_traces.jsonl")
PLOTS_DIR     = Path("results/plots")

CONDITION_LABELS = {
    "original":               "Original",
    "premise_deletion":       "Premise\nDeletion",
    "contradiction_injection":"Contradiction\nInjection",
    "shuffled":               "Shuffled",
}

PERTURBATION_ORDER = ["premise_deletion", "contradiction_injection", "shuffled"]

PALETTE = {
    "gsm8k":      "#4C72B0",
    "strategyqa": "#DD8452",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores() -> pd.DataFrame:
    """Load scored_traces.jsonl into a DataFrame."""
    rows = []
    with open(SCORED_TRACES) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["condition_label"] = df["condition"].map(CONDITION_LABELS)
    df["dataset_label"]   = df["dataset"].str.upper()
    return df


# ---------------------------------------------------------------------------
# Plot 1: Mean CS by condition and dataset (RQ1 + RQ2 overview)
# ---------------------------------------------------------------------------

def plot_cs_by_condition(df: pd.DataFrame):
    """Bar chart of mean CS across all four conditions, grouped by dataset."""
    fig, ax = plt.subplots(figsize=(9, 5))

    all_conditions = ["original"] + PERTURBATION_ORDER
    condition_labels = [CONDITION_LABELS[c] for c in all_conditions]
    x = np.arange(len(all_conditions))
    width = 0.35
    datasets = ["gsm8k", "strategyqa"]

    for i, dataset in enumerate(datasets):
        sub = df[df["dataset"] == dataset]
        means = []
        sems  = []
        for cond in all_conditions:
            vals = sub[sub["condition"] == cond]["cs"].dropna()
            means.append(vals.mean() if len(vals) else 0.0)
            sems.append(vals.sem() if len(vals) > 1 else 0.0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, label=dataset.upper(),
                      color=PALETTE[dataset], alpha=0.85, yerr=sems,
                      capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Mean Coherence Score (CS)", fontsize=12)
    ax.set_title("Mean Toulmin Coherence Score by Condition and Dataset", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Compromised threshold (0.5)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    out = PLOTS_DIR / "cs_by_perturbation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Mean CS_drop by perturbation type and dataset (RQ2)
# ---------------------------------------------------------------------------

def plot_cs_drop_by_type(df: pd.DataFrame):
    """Bar chart of mean CS_drop per perturbation type, grouped by dataset."""
    perturbed = df[df["condition"].isin(PERTURBATION_ORDER) & df["cs_drop"].notna()].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PERTURBATION_ORDER))
    width = 0.35
    datasets = ["gsm8k", "strategyqa"]

    for i, dataset in enumerate(datasets):
        sub = perturbed[perturbed["dataset"] == dataset]
        means = []
        sems  = []
        for cond in PERTURBATION_ORDER:
            vals = sub[sub["condition"] == cond]["cs_drop"].dropna()
            means.append(vals.mean() if len(vals) else 0.0)
            sems.append(vals.sem() if len(vals) > 1 else 0.0)

        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, label=dataset.upper(),
               color=PALETTE[dataset], alpha=0.85, yerr=sems,
               capsize=4, error_kw={"linewidth": 1.2})

    labels = [CONDITION_LABELS[c] for c in PERTURBATION_ORDER]
    ax.set_xlabel("Perturbation Type", fontsize=12)
    ax.set_ylabel("Mean CS Drop (CS_original − CS_perturbed)", fontsize=12)
    ax.set_title("Coherence Score Drop by Perturbation Type (RQ2)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    out = PLOTS_DIR / "cs_drop_by_type.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: CS_drop vs AAD scatter with Pearson r (RQ3)
# ---------------------------------------------------------------------------

def plot_cs_drop_vs_aad(df: pd.DataFrame) -> dict:
    """
    Scatter plot of CS_drop vs accuracy_drop per perturbed condition.
    Computes Pearson r per dataset and overall.
    Returns dict of correlation results.
    """
    perturbed = df[
        df["condition"].isin(PERTURBATION_ORDER) &
        df["cs_drop"].notna() &
        df["accuracy_drop"].notna()
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    correlations = {}

    for ax, dataset in zip(axes, ["gsm8k", "strategyqa"]):
        sub = perturbed[perturbed["dataset"] == dataset]
        x_vals = sub["cs_drop"].values
        y_vals = sub["accuracy_drop"].values.astype(float)

        if len(x_vals) > 1:
            r, p = stats.pearsonr(x_vals, y_vals)
        else:
            r, p = float("nan"), float("nan")

        correlations[dataset] = {"r": r, "p": p, "n": len(x_vals)}

        # Jitter y slightly so overlapping points are visible
        jitter = np.random.default_rng(42).uniform(-0.04, 0.04, len(y_vals))
        ax.scatter(x_vals, y_vals + jitter, alpha=0.45, s=22,
                   color=PALETTE[dataset], edgecolors="none")

        # Regression line
        if len(x_vals) > 1 and not np.isnan(r):
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, m * x_line + b, color="black", linewidth=1.2, linestyle="--")

        p_str = f"p={p:.3f}" if not np.isnan(p) else "p=N/A"
        ax.set_title(f"{dataset.upper()}\nPearson r = {r:.3f}, {p_str} (n={len(x_vals)})",
                     fontsize=11)
        ax.set_xlabel("CS Drop (CS_original − CS_perturbed)", fontsize=10)
        ax.set_ylabel("Accuracy Drop (0 or 1)", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="-")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="-")
        ax.grid(alpha=0.2)
        sns.despine(ax=ax)

    fig.suptitle("CS Drop vs Accuracy Drop: Predictive Correlation (RQ3)", fontsize=13, y=1.01)
    out = PLOTS_DIR / "cs_drop_vs_aad.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return correlations


# ---------------------------------------------------------------------------
# Plot 4: Compromised rate by condition and dataset
# ---------------------------------------------------------------------------

def plot_compromised_rate(df: pd.DataFrame):
    """Bar chart of proportion of Compromised traces per condition per dataset."""
    all_conditions = ["original"] + PERTURBATION_ORDER
    condition_labels = [CONDITION_LABELS[c] for c in all_conditions]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(all_conditions))
    width = 0.35
    datasets = ["gsm8k", "strategyqa"]

    for i, dataset in enumerate(datasets):
        sub = df[df["dataset"] == dataset]
        rates = []
        for cond in all_conditions:
            cond_rows = sub[sub["condition"] == cond]["compromised"].dropna()
            rate = cond_rows.mean() if len(cond_rows) else 0.0
            rates.append(rate)

        offset = (i - 0.5) * width
        ax.bar(x + offset, rates, width, label=dataset.upper(),
               color=PALETTE[dataset], alpha=0.85)

    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Proportion Compromised (CS < 0.5)", fontsize=12)
    ax.set_title("Rate of Compromised Reasoning Traces by Condition", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    out = PLOTS_DIR / "compromised_rate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------

def print_stats(df: pd.DataFrame, correlations: dict):
    """Print a structured summary of all key metrics."""
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    for dataset in ["gsm8k", "strategyqa"]:
        sub = df[df["dataset"] == dataset]
        print(f"\n--- {dataset.upper()} ---")

        # CS by condition
        print("  Mean CS by condition:")
        for cond in ["original"] + PERTURBATION_ORDER:
            vals = sub[sub["condition"] == cond]["cs"].dropna()
            if len(vals):
                print(f"    {cond:<28} {vals.mean():.3f}  (n={len(vals)})")

        # CS_drop by perturbation type
        print("  Mean CS_drop by perturbation type:")
        for cond in PERTURBATION_ORDER:
            vals = sub[sub["condition"] == cond]["cs_drop"].dropna()
            if len(vals):
                print(f"    {cond:<28} {vals.mean():.3f}  (n={len(vals)})")

        # Accuracy
        orig_correct = sub[sub["condition"] == "original"]["is_correct"]
        print(f"  Original accuracy:           {orig_correct.mean():.3f} ({orig_correct.sum()}/{len(orig_correct)})")

        for cond in PERTURBATION_ORDER:
            vals = sub[sub["condition"] == cond]["accuracy_drop"].dropna()
            if len(vals):
                print(f"  AAD ({cond:<22}): {vals.mean():.3f}")

        # RQ3
        c = correlations.get(dataset, {})
        print(f"  Pearson r (CS_drop ~ AAD):   r={c.get('r', float('nan')):.3f}, "
              f"p={c.get('p', float('nan')):.3f}, n={c.get('n', 0)}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scored traces...")
    df = load_scores()
    print(f"Loaded {len(df)} rows across datasets: {df['dataset'].unique().tolist()}")

    print("\nGenerating plots...")
    plot_cs_by_condition(df)
    plot_cs_drop_by_type(df)
    correlations = plot_cs_drop_vs_aad(df)
    plot_compromised_rate(df)

    print_stats(df, correlations)
