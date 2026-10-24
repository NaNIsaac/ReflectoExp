"""
Ablation comparison figures (OriginLab style).

Reads the summary CSVs of the clean and augmented evaluation domains and makes report figures.
Both domains must be drawn together: on clean evaluation alone, variants trained without
augmentation look better (train and eval distributions match), which is not generalization
performance.

Usage:
    python runs/exp07_fourier_hires/plot_ablation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CONFIG
from origin_style import (
    INK,
    ORIGIN_COLORS,
    ORIGIN_MARKERS,
    apply_origin_style,
    style_axes,
)

apply_origin_style()

PARAMS = [
    ("Thickness", "Thickness (Å)", "MAE (Å)"),
    ("Roughness", "Roughness (Å)", "MAE (Å)"),
    ("SLD", r"SLD (10$^{-6}$ Å$^{-2}$)", r"MAE (10$^{-6}$ Å$^{-2}$)"),
]

# only two domains, so only two series colors (fixed order).
# Symbols differ too, so the series are not told apart by color alone.
DOMAIN_STYLE = {
    "clean": {"color": ORIGIN_COLORS[1], "marker": "o", "label": "Clean simulation"},
    "augmented": {"color": ORIGIN_COLORS[0], "marker": "s", "label": "With measurement noise"},
}

CONNECTOR = "#9A9A9A"


def _dumbbell(ax, y, lo, hi):
    """Connector between the two domain values; a guide that shows the shift rather than the values."""
    ax.plot([lo, hi], [y, y], color=CONNECTOR, lw=1.2, zorder=1)


def load_summaries(ablation_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the per-domain summary CSVs; missing ones are skipped silently."""
    candidates = {
        "clean": ["ablation_summary_clean.csv", "ablation_summary.csv"],
        "augmented": ["ablation_summary_augmented.csv"],
    }
    out = {}
    for domain, names in candidates.items():
        for fname in names:
            path = ablation_dir / fname
            if path.exists():
                out[domain] = pd.read_csv(path)
                print(f"[Load] {domain:<9} <- {fname}")
                break
        else:
            print(f"[Skip] {domain}: summary CSV not found")
    return out


def plot_mae_by_variant(dfs: dict[str, pd.DataFrame], save_path: Path):
    """
    Per-parameter MAE as a variant x evaluation-domain dumbbell dot plot.

    Why not bars: the values span more than two decades and need a log axis, and bar length on a
    log axis is not proportional to magnitude, so it misreads.
    """
    if not dfs:
        return

    order = list(next(iter(dfs.values()))["variant"])
    domains = [d for d in ("clean", "augmented") if d in dfs]
    y = np.arange(len(order))[::-1]  # first variant on top

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4.9 * len(PARAMS), 4.2), sharey=True)

    for ax, (key, title, xlabel) in zip(axes, PARAMS, strict=True):
        style_axes(ax)
        col = f"{key}_MAE"
        series = {d: dfs[d].set_index("variant") for d in domains}

        if len(domains) == 2:
            for yi, v in zip(y, order, strict=True):
                lo = series["clean"].loc[v, col]
                hi = series["augmented"].loc[v, col]
                _dumbbell(ax, yi, lo, hi)

        for domain in domains:
            df = series[domain]
            vals = [df.loc[v, col] if v in df.index else np.nan for v in order]
            ax.plot(
                vals,
                y,
                ls="none",
                marker=DOMAIN_STYLE[domain]["marker"],
                ms=8,
                mfc=DOMAIN_STYLE[domain]["color"],
                mec=INK,
                mew=0.8,
                label=DOMAIN_STYLE[domain]["label"],
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="y", which="minor", left=False, right=False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)

    # placed horizontally above the figure; inside the panel it would cover data points
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Ablation: mean absolute error by variant and evaluation domain", fontsize=13, y=1.06)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Save] {save_path.name}")


def plot_domain_gap(dfs: dict[str, pd.DataFrame], save_path: Path):
    """
    How many times the error grows when the domain changes from clean to augmented.
    The most direct measure of the effect of augmentation.
    """
    if not {"clean", "augmented"} <= dfs.keys():
        print("[Skip] domain gap plot: needs both clean and augmented summaries")
        return

    clean = dfs["clean"].set_index("variant")
    aug = dfs["augmented"].set_index("variant")
    order = [v for v in clean.index if v in aug.index]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    style_axes(ax)

    y = np.arange(len(order))[::-1]
    offset = 0.18

    for i, (key, title, _) in enumerate(PARAMS):
        col = f"{key}_MAE"
        ratio = [aug.loc[v, col] / clean.loc[v, col] for v in order]
        ax.plot(
            ratio,
            y + (i - (len(PARAMS) - 1) / 2) * offset,
            ls="none",
            marker=ORIGIN_MARKERS[i],
            ms=8,
            mfc=ORIGIN_COLORS[i],
            mec=INK,
            mew=0.8,
            label=title,
            zorder=3,
        )

    ax.axvline(1.0, color=INK, linestyle="--", lw=1.2)
    ax.text(1.05, len(order) - 0.6, "no degradation", va="top", ha="left", fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("MAE ratio (with noise / clean)")
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_title("Robustness to measurement noise (lower is better)")
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Save] {save_path.name}")


def plot_r2_panel(dfs: dict[str, pd.DataFrame], save_path: Path):
    """R2 sits near 1, so 1 - R2 is drawn on a log axis to make differences visible."""
    if not dfs:
        return

    order = list(next(iter(dfs.values()))["variant"])
    domains = [d for d in ("clean", "augmented") if d in dfs]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    style_axes(ax)

    y = np.arange(len(order))[::-1]
    series = {d: dfs[d].set_index("variant") for d in domains}

    if len(domains) == 2:
        for yi, v in zip(y, order, strict=True):
            _dumbbell(ax, yi, 1.0 - series["clean"].loc[v, "Thickness_R2"], 1.0 - series["augmented"].loc[v, "Thickness_R2"])

    for domain in domains:
        df = series[domain]
        vals = [1.0 - df.loc[v, "Thickness_R2"] if v in df.index else np.nan for v in order]
        ax.plot(
            vals,
            y,
            ls="none",
            marker=DOMAIN_STYLE[domain]["marker"],
            ms=8,
            mfc=DOMAIN_STYLE[domain]["color"],
            mec=INK,
            mew=0.8,
            label=DOMAIN_STYLE[domain]["label"],
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(r"$1 - R^2$  (thickness)")
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_title("Unexplained variance in thickness (lower is better)")
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    # the top two rows have small x, so the upper right is empty (the bottom row is taken by E's noise values)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Save] {save_path.name}")


def main():
    root = Path(CONFIG["base_dir"]) / CONFIG["exp_name"]
    ablation_dir = root / "ablation"
    if not ablation_dir.exists():
        raise SystemExit(f"Ablation directory not found: {ablation_dir}")

    dfs = load_summaries(ablation_dir)
    if not dfs:
        raise SystemExit("No summary CSV found - run ablation.py first")

    fig_dir = ablation_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_mae_by_variant(dfs, fig_dir / "ablation_mae.png")
    plot_r2_panel(dfs, fig_dir / "ablation_r2.png")
    plot_domain_gap(dfs, fig_dir / "ablation_domain_gap.png")

    print(f"\n✅ Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
