"""
Chinchilla-style plots for the sweep.

Two charts:
  1. IsoFLOPs curves (val loss vs params, one curve per compute budget) — the
     U-shape tells you the compute-optimal model size per budget.
  2. Compute frontier (best val loss vs budget) — shows diminishing returns
     so you can decide how long to pedal.

    python plot_sweep.py
    python plot_sweep.py path/to/results.json
"""

import json
import os
import sys

import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "monospace", "font.size": 10})

BUDGET_COLOURS = {30: "#2b8cbe", 120: "#e6550d", 240: "#31a354"}


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_isoflops(results, out):
    budgets = sorted({r["budget_minutes"] for r in results})
    fig, ax = plt.subplots(figsize=(7, 5))

    best_per_budget = {}
    for bm in budgets:
        group = sorted(
            [r for r in results if r["budget_minutes"] == bm],
            key=lambda r: r["n_params"],
        )
        xs = [r["n_params"] for r in group]
        ys = [r["best_val_loss"] for r in group]
        c = BUDGET_COLOURS.get(bm, "#666")
        ax.plot(xs, ys, color=c, marker="o", lw=1.8, ms=6, label=f"{bm} min")

        best = min(group, key=lambda r: r["best_val_loss"])
        best_per_budget[bm] = best
        ax.plot(
            best["n_params"],
            best["best_val_loss"],
            marker="*",
            color=c,
            ms=16,
            mec="black",
            mew=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{best['n_params']//1000}k",
            (best["n_params"], best["best_val_loss"]),
            textcoords="offset points",
            xytext=(8, 6),
            color=c,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Best validation loss")
    ax.set_title(
        "IsoFLOPs: optimal model size per compute budget\n"
        "(★ = Chinchilla-optimal point on each curve)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Bike time", loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_frontier(results, out):
    budgets = sorted({r["budget_minutes"] for r in results})
    winners = []
    for bm in budgets:
        group = [r for r in results if r["budget_minutes"] == bm]
        winners.append(min(group, key=lambda r: r["best_val_loss"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [w["budget_minutes"] for w in winners]
    ys = [w["best_val_loss"] for w in winners]
    ax.plot(xs, ys, color="#333", marker="o", lw=2, ms=9)

    for w in winners:
        ax.annotate(
            f"{w['n_params']//1000}k params",
            (w["budget_minutes"], w["best_val_loss"]),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=9,
        )

    ax.set_xlabel("Bike time (minutes on a Pi at 1.72 GFLOPs/s)")
    ax.set_ylabel("Best validation loss  (compute-optimal model)")
    ax.set_title("Compute frontier: diminishing returns")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(budgets)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", "sweep", "results.json"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(path):
        print(f"results.json not found at {path}")
        sys.exit(1)

    results = _load(path)
    out_dir = os.path.dirname(path)
    print(f"Loaded {len(results)} configs from {path}")

    plot_isoflops(results, os.path.join(out_dir, "isoflops.png"))
    plot_frontier(results, os.path.join(out_dir, "frontier.png"))
