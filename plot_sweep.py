"""
Visualise Chinchilla-style sweep results.

Usage:
    python plot_sweep.py                        # reads logs/sweep/results.json
    python plot_sweep.py path/to/results.json
"""

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#3a3d4d",
        "axes.labelcolor": "#c8ccd8",
        "axes.titlecolor": "#e8ecf4",
        "axes.grid": True,
        "grid.color": "#2a2d3d",
        "grid.linewidth": 0.6,
        "xtick.color": "#8890a8",
        "ytick.color": "#8890a8",
        "text.color": "#c8ccd8",
        "legend.facecolor": "#1a1d27",
        "legend.edgecolor": "#3a3d4d",
        "legend.labelcolor": "#c8ccd8",
        "font.family": "monospace",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.titlesize": 13,
    }
)

# Colour per compute budget
_BUDGET_COLOURS = {30: "#5b8dee", 60: "#43d9ad", 120: "#f7c35f", 180: "#f76e6e"}
# Marker per data fraction
_FRAC_MARKER = {0.25: "^", 0.5: "s", 1.0: "o"}
_FRAC_ALPHA = {0.25: 0.50, 0.5: 0.72, 1.0: 1.00}
_FRAC_DASH = {0.25: (4, 3), 0.5: (2, 2), 1.0: None}


def _budget_colour(r):
    return _BUDGET_COLOURS.get(r["budget_minutes"], "#aaaaaa")


# ── helpers ───────────────────────────────────────────────────────────────────
def _budgets(results):
    return sorted(set(r["budget_minutes"] for r in results))


def _params(results):
    return sorted(set(r["n_params"] for r in results))


def _fracs(results):
    return sorted(set(r["config"]["data_fraction"] for r in results))


def _budget_legend(ax):
    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=f"{bmin} min")
        for bmin, c in sorted(_BUDGET_COLOURS.items())
    ]
    ax.legend(handles=handles, fontsize=8)


def _frac_legend(ax, loc="upper right"):
    handles = [
        plt.Line2D(
            [0], [0], color="#aaaaaa", lw=1.5, alpha=_FRAC_ALPHA[f],
            dashes=_FRAC_DASH[f] or (None, None),
            marker=_FRAC_MARKER[f], ms=6,
            label=f"data={int(f * 100)}%",
        )
        for f in [0.25, 0.5, 1.0]
    ]
    ax.legend(handles=handles, fontsize=8, loc=loc)


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.relpath(path)}")


# ── plot 1: val loss vs params, one curve per budget, one panel per data% ─────
def plot_param_scaling(results, out_dir):
    fracs = _fracs(results)
    budgets = _budgets(results)

    fig, axes = plt.subplots(1, len(fracs), figsize=(5 * len(fracs), 5), sharey=True)
    if len(fracs) == 1:
        axes = [axes]
    fig.suptitle("Val Loss vs Params — one curve per compute budget", fontweight="bold")

    for ax, frac in zip(axes, fracs):
        for bmin in budgets:
            group = sorted(
                [r for r in results
                 if r["budget_minutes"] == bmin
                 and abs(r["config"]["data_fraction"] - frac) < 0.01],
                key=lambda r: r["n_params"],
            )
            if not group:
                continue
            xs = [r["n_params"] for r in group]
            ys = [r["best_val_loss"] for r in group]
            col = _BUDGET_COLOURS.get(bmin, "#aaaaaa")
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6, label=f"{bmin} min")
            # annotate best point
            best = min(group, key=lambda r: r["best_val_loss"])
            ax.annotate(
                f"{best['n_params'] // 1000}k",
                (best["n_params"], best["best_val_loss"]),
                textcoords="offset points", xytext=(5, -12),
                fontsize=7, color=col,
            )

        ax.set_title(f"data={int(frac * 100)}%")
        ax.set_xlabel("Parameters")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x) // 1000}k"))
        if ax is axes[0]:
            ax.set_ylabel("Best val loss")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, "param_scaling.png")


# ── plot 2: best val loss vs compute budget — diminishing returns ──────────────
def plot_budget_frontier(results, out_dir):
    budgets = _budgets(results)
    fracs = _fracs(results)
    params_list = _params(results)

    fig, axes = plt.subplots(1, len(fracs), figsize=(5 * len(fracs), 5), sharey=True)
    if len(fracs) == 1:
        axes = [axes]
    fig.suptitle(
        "Diminishing Returns: Best Val Loss vs Compute Budget",
        fontweight="bold",
    )

    param_colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(params_list)))

    for ax, frac in zip(axes, fracs):
        for pidx, np_ in enumerate(params_list):
            pts = sorted(
                [r for r in results
                 if r["n_params"] == np_
                 and abs(r["config"]["data_fraction"] - frac) < 0.01],
                key=lambda r: r["budget_minutes"],
            )
            if not pts:
                continue
            xs = [r["budget_minutes"] for r in pts]
            ys = [r["best_val_loss"] for r in pts]
            col = param_colours[pidx]
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6,
                    label=f"{np_ // 1000}k params")

        ax.set_title(f"data={int(frac * 100)}%")
        ax.set_xlabel("Budget (Pi wall-clock minutes)")
        ax.set_xticks(budgets)
        if ax is axes[0]:
            ax.set_ylabel("Best val loss")
        ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, out_dir, "budget_frontier.png")


# ── plot 3: data fraction effect — val loss vs data%, one curve per budget ────
def plot_data_scaling(results, out_dir):
    budgets = _budgets(results)
    params_list = _params(results)

    fig, axes = plt.subplots(1, len(params_list), figsize=(4.5 * len(params_list), 5), sharey=True)
    if len(params_list) == 1:
        axes = [axes]
    fig.suptitle(
        "Data Diversity Effect: Val Loss vs Dataset Fraction\n"
        "(tokens_seen held constant within each budget cell)",
        fontweight="bold",
    )

    fracs = _fracs(results)
    x_ticks = [int(f * 100) for f in fracs]

    for ax, np_ in zip(axes, params_list):
        for bmin in budgets:
            group = sorted(
                [r for r in results
                 if r["budget_minutes"] == bmin and r["n_params"] == np_],
                key=lambda r: r["config"]["data_fraction"],
            )
            if not group:
                continue
            xs = [int(r["config"]["data_fraction"] * 100) for r in group]
            ys = [r["best_val_loss"] for r in group]
            col = _BUDGET_COLOURS.get(bmin, "#aaaaaa")
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6, label=f"{bmin} min")

        ax.set_title(f"{np_ // 1000}k params")
        ax.set_xlabel("Dataset fraction (%)")
        ax.set_xticks(x_ticks)
        if ax is axes[0]:
            ax.set_ylabel("Best val loss")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, "data_scaling.png")


# ── plot 4: loss curves vs training step, grid by budget ──────────────────────
def plot_loss_curves(results, out_dir):
    budgets = _budgets(results)
    params_list = _params(results)

    fig, axes = plt.subplots(
        len(budgets), len(params_list),
        figsize=(4.5 * len(params_list), 4 * len(budgets)),
        sharex=False, sharey=False,
    )
    if len(budgets) == 1:
        axes = [axes]
    if len(params_list) == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle("Loss Curves by Budget × Model Size  (colour = data fraction)", fontweight="bold", y=1.01)

    frac_cols = {0.25: "#f76e6e", 0.5: "#f7c35f", 1.0: "#43d9ad"}

    for row, bmin in enumerate(budgets):
        for col, np_ in enumerate(params_list):
            ax = axes[row][col]
            group = [r for r in results if r["budget_minutes"] == bmin and r["n_params"] == np_]
            for r in group:
                frac = r["config"]["data_fraction"]
                col_c = frac_cols.get(frac, "#aaaaaa")
                hist = r["loss_history"]
                if not hist:
                    continue
                steps = [e["step"] for e in hist]
                ax.plot(steps, [e["train_loss"] for e in hist], color=col_c, alpha=0.4, lw=1.2)
                ax.plot(steps, [e["val_loss"] for e in hist], color=col_c, alpha=1.0, lw=1.8,
                        label=f"d{int(frac * 100)}%")
            ax.set_title(f"{bmin}m / {np_ // 1000}k params", fontsize=8)
            ax.set_xlabel("step", fontsize=7)
            ax.set_ylabel("loss", fontsize=7)
            ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    _save(fig, out_dir, "loss_curves.png")


# ── plot 5: loss vs cumulative GFLOPs (training FLOPs only) ──────────────────
def plot_loss_vs_compute(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Loss vs Cumulative Training GFLOPs", fontweight="bold")

    for r in results:
        hist = r["loss_history"]
        if not hist:
            continue
        frac = r["config"]["data_fraction"]
        gf = [e["cumul_gflops"] for e in hist]
        col = _BUDGET_COLOURS.get(r["budget_minutes"], "#aaaaaa")
        al = _FRAC_ALPHA[frac]
        dash = _FRAC_DASH[frac]
        kw = dict(color=col, alpha=al, lw=1.2)
        if dash:
            kw["dashes"] = dash
        axes[0].plot(gf, [e["train_loss"] for e in hist], **kw)
        axes[1].plot(gf, [e["val_loss"] for e in hist], **kw)

    for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
        ax.set_xlabel("Cumulative Training GFLOPs")
        ax.set_ylabel("Loss")
        ax.set_title(title)

    _budget_legend(axes[0])
    _frac_legend(axes[1])

    fig.tight_layout()
    _save(fig, out_dir, "loss_vs_compute.png")


# ── plot 6: heatmap grid — budget × params, one panel per data fraction ───────
def plot_heatmap(results, out_dir):
    budgets = _budgets(results)
    params_list = _params(results)
    fracs = _fracs(results)

    fig, axes = plt.subplots(1, len(fracs), figsize=(5 * len(fracs), 4))
    if len(fracs) == 1:
        axes = [axes]
    fig.suptitle("Best Val Loss Heatmap  (budget × params)", fontweight="bold")

    for ax, frac in zip(axes, fracs):
        matrix = np.full((len(budgets), len(params_list)), np.nan)
        for i, bmin in enumerate(budgets):
            for j, np_ in enumerate(params_list):
                match = [r for r in results
                         if r["budget_minutes"] == bmin
                         and r["n_params"] == np_
                         and abs(r["config"]["data_fraction"] - frac) < 0.01]
                if match:
                    matrix[i, j] = match[0]["best_val_loss"]

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="best val loss")

        ax.set_xticks(range(len(params_list)))
        ax.set_xticklabels([f"{p // 1000}k" for p in params_list])
        ax.set_yticks(range(len(budgets)))
        ax.set_yticklabels([f"{b}m" for b in budgets])
        ax.set_xlabel("Params")
        ax.set_ylabel("Budget")
        ax.set_title(f"data={int(frac * 100)}%")

        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
        for i in range(len(budgets)):
            for j in range(len(params_list)):
                v = matrix[i, j]
                if not math.isnan(v):
                    brightness = (v - vmin) / max(vmax - vmin, 1e-9)
                    txt_col = "#0f1117" if brightness < 0.6 else "#e8ecf4"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color=txt_col, fontweight="bold")

    fig.tight_layout()
    _save(fig, out_dir, "heatmap.png")


# ── plot 7: overfitting gap vs repetitions ────────────────────────────────────
def plot_overfitting(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Overfitting: Val−Train Gap vs Data Repetitions", fontweight="bold")

    for r in results:
        gap = r.get("val_train_gap")
        rep = r.get("repetitions")
        if gap is None or rep is None:
            continue
        col = _BUDGET_COLOURS.get(r["budget_minutes"], "#aaaaaa")
        frac = r["config"]["data_fraction"]
        ax.scatter(rep, gap, color=col, marker=_FRAC_MARKER[frac], s=60, alpha=0.85, zorder=3)

    ax.axhline(0, color="#3a3d4d", lw=1)
    ax.set_xlabel("Repetitions  (tokens_seen / unique_train_tokens)")
    ax.set_ylabel("Val loss − Train loss")
    _budget_legend(ax)
    _frac_legend(ax, loc="upper left")

    fig.tight_layout()
    _save(fig, out_dir, "overfitting_gap.png")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))

    results_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(_dir, "logs", "sweep", "results.json")

    if not os.path.exists(results_path):
        print(f"results.json not found at {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    out_dir = os.path.dirname(results_path)
    print(f"Loaded {len(results)} configs from {results_path}")
    print(f"Writing plots to {out_dir}/\n")

    plot_param_scaling(results, out_dir)
    plot_budget_frontier(results, out_dir)
    plot_data_scaling(results, out_dir)
    plot_loss_curves(results, out_dir)
    plot_loss_vs_compute(results, out_dir)
    plot_heatmap(results, out_dir)
    plot_overfitting(results, out_dir)

    print("\nDone.")