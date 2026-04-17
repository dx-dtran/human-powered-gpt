"""
Chinchilla-style parallel hyperparameter sweep.

Grid: 4 compute budgets × 4 model sizes × 3 dataset fractions = 48 configs.

Within each (budget, model_size) cell, max_iters is derived so that
wall-clock time on the Pi ≈ budget target INCLUDING eval overhead.
Dataset fraction is then a pure data-diversity axis: tokens_seen is held
constant across fractions within a cell.

Usage:
    python sweep.py            # full 48-config sweep
    python sweep.py --dry-run  # 20 iters per config, validates pipeline
"""

import concurrent.futures
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from train import (
    GPT,
    GPTConfig,
    Tokenizer,
    create_vocabulary,
    estimate_loss,
    get_batch,
    get_data,
    get_train_val_data,
    _fmt_flops,
)

# ── constants ─────────────────────────────────────────────────────────────────
# Pi throughput from reference run on commit e06b38b (2026-04-16):
#   5.32 TF training in 3143.6s total; window-based throughput (pure training,
#   eval excluded) was 1.72–1.83 GFLOPs/s. Using 1.72 as conservative estimate.
#   Note: architecture was rewritten in c83d4c4 — actual Pi throughput may differ.
PI_GFLOPS_S = 1.72

BUDGET_MINUTES = [30, 60, 120, 180]

BATCH_SIZE = 64
CONTEXT_LENGTH = 64
LEARNING_RATE = 3e-3
LR_MIN = 1e-4
TRAIN_VAL_SPLIT = 0.9

# Eval settings — kept lean to limit eval overhead on Pi.
EVAL_ITERS = 10       # batches averaged per loss estimate (train + val)
N_CHECKPOINTS = 10    # number of eval checkpoints per training run

DATA_PATH = os.path.join(_DIR, "dataset.txt")
LOGS_DIR = os.path.join(_DIR, "logs", "sweep")


# ── config dataclass ──────────────────────────────────────────────────────────
@dataclass
class SweepConfig:
    label: str
    budget_label: str       # e.g. "30m"
    budget_minutes: int
    n_layer: int
    data_fraction: float
    max_iters: int
    n_embd: int = 64
    n_head: int = 4
    batch_size: int = BATCH_SIZE
    context_length: int = CONTEXT_LENGTH
    seed: int = 42
    log_path: str = ""


# ── worker ────────────────────────────────────────────────────────────────────
def run_config(cfg: SweepConfig) -> dict:
    """Train one config. Module-level so ProcessPoolExecutor can pickle it."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    device = "cpu"

    raw_data = get_data(DATA_PATH)
    vocab, vocab_size = create_vocabulary(raw_data)
    tokenizer = Tokenizer(vocab)
    train_data, val_data = get_train_val_data(raw_data, tokenizer, device, TRAIN_VAL_SPLIT)

    # Slice training corpus to the requested fraction; val always uses full split.
    n_train = max(cfg.context_length + 2, int(len(train_data) * cfg.data_fraction))
    train_data = train_data[:n_train]

    model = GPT(
        GPTConfig(
            block_size=cfg.context_length,
            vocab_size=vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=0.0,
            bias=True,
        )
    )
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = 6 * n_params * cfg.batch_size * cfg.context_length

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    eval_interval = max(cfg.max_iters // N_CHECKPOINTS, 1)
    loss_history = []
    cumulative_flops = 0.0
    window_flops = 0.0
    window_start = time.time()
    start = time.time()

    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    with open(cfg.log_path, "w", buffering=1) as lf:
        lf.write(f"=== run: {cfg.label} ===\n")
        lf.write(f"label           : {cfg.label}\n")
        lf.write(f"budget          : {cfg.budget_label} ({cfg.budget_minutes} min)\n")
        lf.write(f"params          : {n_params:,}\n")
        lf.write(f"n_embd          : {cfg.n_embd}\n")
        lf.write(f"n_layer         : {cfg.n_layer}\n")
        lf.write(f"n_head          : {cfg.n_head}\n")
        lf.write(f"data_fraction   : {cfg.data_fraction}\n")
        lf.write(f"train_tokens    : {n_train:,}\n")
        lf.write(f"val_tokens      : {len(val_data):,}\n")
        lf.write(f"max_iters       : {cfg.max_iters}\n")
        lf.write(f"eval_interval   : {eval_interval}\n")
        lf.write(f"eval_iters      : {EVAL_ITERS}\n")
        lf.write(f"batch_size      : {cfg.batch_size}\n")
        lf.write(f"context_length  : {cfg.context_length}\n")
        lf.write(f"seed            : {cfg.seed}\n")
        lf.write(f"flops_per_step  : {flops_per_step:,}\n")
        lf.write(f"device          : {device}\n")
        lf.write(f"started_at      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write("\n")

        col_hdr = (
            f"{'step':>6}  {'train_loss':>10}  {'val_loss':>9}  "
            f"{'lr':>9}  {'elapsed_s':>9}  {'cumul_gflops':>12}  "
            f"{'gflops_per_s':>12}  {'eta_s':>7}\n"
        )
        lf.write(col_hdr)
        lf.write("-" * len(col_hdr.rstrip()) + "\n")

        for step in range(cfg.max_iters):
            progress = step / max(cfg.max_iters - 1, 1)
            lr = LR_MIN + 0.5 * (LEARNING_RATE - LR_MIN) * (
                1 + math.cos(math.pi * progress)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y = get_batch(train_data, cfg.batch_size, cfg.context_length)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_flops += flops_per_step
            window_flops += flops_per_step

            if step % eval_interval == 0 or step == cfg.max_iters - 1:
                now = time.time()
                elapsed = now - start
                window_elapsed = max(now - window_start, 1e-9)
                gflops_s = window_flops / window_elapsed / 1e9
                cumul_gf = cumulative_flops / 1e9

                train_loss = estimate_loss(
                    model, train_data, cfg.batch_size, cfg.context_length, EVAL_ITERS
                ).item()
                val_loss = estimate_loss(
                    model, val_data, cfg.batch_size, cfg.context_length, EVAL_ITERS
                ).item()

                steps_done = step + 1
                eta_s = int((cfg.max_iters - steps_done) * elapsed / steps_done) if steps_done > 1 else -1

                entry = {
                    "step": step,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "lr": round(lr, 8),
                    "elapsed_s": round(elapsed, 3),
                    "cumul_gflops": round(cumul_gf, 4),
                    "gflops_per_s": round(gflops_s, 4),
                }
                loss_history.append(entry)

                lf.write(
                    f"{step:>6}  {train_loss:>10.6f}  {val_loss:>9.6f}  "
                    f"{lr:>9.6f}  {elapsed:>9.2f}  {cumul_gf:>12.4f}  "
                    f"{gflops_s:>12.4f}  {eta_s:>7}\n"
                )

                window_start = time.time()
                window_flops = 0.0

        total_elapsed = time.time() - start
        lf.write(f"\ntotal_elapsed_s : {total_elapsed:.3f}\n")
        lf.write(f"total_gflops    : {cumulative_flops / 1e9:.4f}\n")
        lf.write(f"total_flops_fmt : {_fmt_flops(cumulative_flops)}\n")

    tokens_seen = cfg.max_iters * cfg.batch_size * cfg.context_length
    final_train = loss_history[-1]["train_loss"] if loss_history else None
    final_val = loss_history[-1]["val_loss"] if loss_history else None
    best_val = min(e["val_loss"] for e in loss_history) if loss_history else None
    return {
        "label": cfg.label,
        "budget_label": cfg.budget_label,
        "budget_minutes": cfg.budget_minutes,
        "config": asdict(cfg),
        "n_params": n_params,
        "vocab_size": vocab_size,
        "n_train_tokens": n_train,
        "n_val_tokens": len(val_data),
        "tokens_seen": tokens_seen,
        "repetitions": round(tokens_seen / max(n_train, 1), 2),
        "flops_per_step": flops_per_step,
        "loss_history": loss_history,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "val_train_gap": round(final_val - final_train, 6) if (final_val is not None and final_train is not None) else None,
        "total_flops": cumulative_flops,
        "elapsed_s": total_elapsed,
    }


# ── sweep config builder ──────────────────────────────────────────────────────
def _param_count(n_layer, n_embd=64, vocab_size=35, block_size=64):
    """Exact parameter count matching GPT model in train.py."""
    per_block = (
        3 * n_embd * n_embd + 3 * n_embd   # c_attn weight + bias
        + n_embd * n_embd + n_embd          # c_proj weight + bias
        + 4 * n_embd * n_embd + 4 * n_embd # mlp c_fc weight + bias
        + 4 * n_embd * n_embd + n_embd      # mlp c_proj weight + bias
        + 2 * n_embd                        # ln_1 weight + bias
        + 2 * n_embd                        # ln_2 weight + bias
    )
    return vocab_size * n_embd + block_size * n_embd + n_layer * per_block + 2 * n_embd


def _derive_max_iters(budget_s: float, n_params: int) -> int:
    """
    Derive max_iters so that total Pi wall-clock time ≈ budget_s.

    Wall time = train_time + eval_time
    eval_time  = N_CHECKPOINTS × 2 × EVAL_ITERS × forward_flops_per_batch / throughput
               = N_CHECKPOINTS × 2 × EVAL_ITERS × (2 × n_params × B × T) / throughput
    train_time = max_iters × 6 × n_params × B × T / throughput

    Solving for max_iters:
      budget_flops = budget_s × PI_GFLOPS_S × 1e9
      eval_flops   = N_CHECKPOINTS × 2 × EVAL_ITERS × 2 × n_params × B × T
      train_flops  = budget_flops - eval_flops
      max_iters    = train_flops / (6 × n_params × B × T)
    """
    batch_tokens = BATCH_SIZE * CONTEXT_LENGTH
    budget_flops = budget_s * PI_GFLOPS_S * 1e9
    eval_flops = N_CHECKPOINTS * 2 * EVAL_ITERS * 2 * n_params * batch_tokens
    train_flops = budget_flops - eval_flops
    if train_flops <= 0:
        return 50  # eval overhead exceeds budget — return minimum
    return max(50, int(train_flops / (6 * n_params * batch_tokens)))


def build_configs(dry_run: bool = False) -> list:
    """Build 4 budgets × 4 model sizes × 3 data fractions = 48 configs."""
    LAYERS = [1, 2, 4, 6]
    FRACS = [0.25, 0.5, 1.0]

    configs = []
    config_id = 0
    for b_idx, budget_min in enumerate(BUDGET_MINUTES):
        budget_label = f"{budget_min}m"
        budget_s = budget_min * 60
        for l_idx, n_layer in enumerate(LAYERS):
            n_params = _param_count(n_layer)
            max_iters = 20 if dry_run else _derive_max_iters(budget_s, n_params)
            params_k = n_params // 1000
            for f_idx, frac in enumerate(FRACS):
                label = f"{budget_label}_{params_k}k_d{int(frac * 100)}"
                log_path = os.path.join(LOGS_DIR, f"{label}.log")
                configs.append(
                    SweepConfig(
                        label=label,
                        budget_label=budget_label,
                        budget_minutes=budget_min,
                        n_layer=n_layer,
                        data_fraction=frac,
                        max_iters=max_iters,
                        seed=42 + config_id,
                        log_path=log_path,
                    )
                )
                config_id += 1
    return configs


# ── parallel executor ─────────────────────────────────────────────────────────
def sweep(configs: list, max_workers: int) -> list:
    total = len(configs)
    results = []
    completed = 0

    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx
    ) as executor:
        future_to_cfg = {executor.submit(run_config, cfg): cfg for cfg in configs}
        for future in concurrent.futures.as_completed(future_to_cfg):
            cfg = future_to_cfg[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{completed:2d}/{total}]  {cfg.label:<22}  "
                    f"train={result['final_train_loss']:.4f}  "
                    f"val={result['final_val_loss']:.4f}  "
                    f"reps={result['repetitions']:.1f}×  "
                    f"{result['elapsed_s'] / 60:.1f} min  "
                    f"log → {os.path.relpath(cfg.log_path)}"
                )
            except Exception as exc:
                import traceback
                print(f"[{completed:2d}/{total}]  FAILED {cfg.label}: {exc}")
                traceback.print_exc()

    results.sort(key=lambda r: (
        r["budget_minutes"], r["n_params"], r["config"]["data_fraction"]
    ))
    return results


# ── reporting ─────────────────────────────────────────────────────────────────
def print_summary_table(results: list) -> None:
    header = (
        f"{'label':<22} {'params':>8} {'data%':>6} {'iters':>6} "
        f"{'tok_seen':>10} {'reps':>6} {'train':>8} {'val':>8} "
        f"{'best_val':>8} {'gap':>7} {'actual_min':>10}"
    )
    bar = "=" * len(header)
    print(bar)
    print("SWEEP RESULTS")
    print(bar)

    current_budget = None
    for r in results:
        if r["budget_label"] != current_budget:
            current_budget = r["budget_label"]
            print(f"\n── Budget: {current_budget} ({'target ' + str(r['budget_minutes']) + ' min on Pi'}) ──")
            print(header)
            print("-" * len(header))
        cfg = r["config"]
        print(
            f"{r['label']:<22} {r['n_params']:>8,} "
            f"{int(cfg['data_fraction'] * 100):>5}% "
            f"{cfg['max_iters']:>6} "
            f"{r['tokens_seen']:>10,} {r['repetitions']:>6.1f} "
            f"{r['final_train_loss']:>8.4f} {r['final_val_loss']:>8.4f} "
            f"{r['best_val_loss']:>8.4f} {r['val_train_gap']:>+7.4f} "
            f"{r['elapsed_s'] / 60:>10.1f}"
        )
    print(bar)
    print(
        "  reps = tokens_seen / unique_train_tokens  |  "
        "gap = val−train (large positive → overfitting)"
    )


def print_budget_winners(results: list) -> None:
    print()
    print("=" * 70)
    print("OPTIMAL ALLOCATION PER COMPUTE BUDGET")
    print("=" * 70)
    print(
        "Within each budget, the best val loss identifies the optimal\n"
        "(params, tokens_seen) allocation. Across budgets, diminishing\n"
        "returns signal the right budget to use for a final training run.\n"
    )

    budgets = sorted(set(r["budget_minutes"] for r in results))
    overall_best = None
    overall_best_val = float("inf")

    prev_best_val = None
    for bmin in budgets:
        group = [r for r in results if r["budget_minutes"] == bmin]
        best = min(group, key=lambda r: r["best_val_loss"])
        improvement = (
            f"  Δ={prev_best_val - best['best_val_loss']:+.4f} vs prev budget"
            if prev_best_val is not None else ""
        )
        print(
            f"  {bmin:>3} min  →  winner: {best['label']:<22}  "
            f"best_val={best['best_val_loss']:.4f}  "
            f"params={best['n_params']:,}  "
            f"tok_seen={best['tokens_seen']:,}{improvement}"
        )
        prev_best_val = best["best_val_loss"]
        if best["best_val_loss"] < overall_best_val:
            overall_best_val = best["best_val_loss"]
            overall_best = best

    print()
    print("=" * 70)
    print("RECOMMENDATION FOR FINAL TRAINING RUN")
    print("=" * 70)
    cfg = overall_best["config"]
    print(f"  label         : {overall_best['label']}")
    print(f"  n_layer       : {cfg['n_layer']}")
    print(f"  n_embd        : {cfg['n_embd']}")
    print(f"  n_params      : {overall_best['n_params']:,}")
    print(f"  max_iters     : {cfg['max_iters']:,}")
    print(f"  data_fraction : {cfg['data_fraction']}")
    print(f"  tokens_seen   : {overall_best['tokens_seen']:,}")
    print(f"  repetitions   : {overall_best['repetitions']:.1f}×")
    print(f"  best_val_loss : {overall_best['best_val_loss']:.4f}")
    print(f"  budget        : {overall_best['budget_label']}")


def print_param_scaling_per_budget(results: list) -> None:
    print()
    print("=" * 70)
    print("PARAMETER SCALING WITHIN EACH BUDGET  (data=100%)")
    print("=" * 70)
    budgets = sorted(set(r["budget_minutes"] for r in results))
    for bmin in budgets:
        full = sorted(
            [r for r in results
             if r["budget_minutes"] == bmin
             and abs(r["config"]["data_fraction"] - 1.0) < 0.01],
            key=lambda r: r["n_params"],
        )
        if not full:
            continue
        print(f"\n── {bmin} min ──────────────────────────────────────────────")
        print(f"  {'params':>8}  {'iters':>6}  {'tok_seen':>10}  {'reps':>5}  {'best_val':>8}  {'gap':>7}")
        for r in full:
            gap = r["val_train_gap"] or 0
            print(
                f"  {r['n_params']:>8,}  {r['config']['max_iters']:>6}  "
                f"{r['tokens_seen']:>10,}  {r['repetitions']:>5.1f}  "
                f"{r['best_val_loss']:>8.4f}  {gap:>+7.4f}"
            )


def save_csv_loss_history(results: list) -> str:
    path = os.path.join(LOGS_DIR, "loss_history.csv")
    with open(path, "w") as f:
        f.write("label,budget_label,budget_minutes,n_params,data_fraction,"
                "step,train_loss,val_loss,lr,elapsed_s,cumul_gflops,gflops_per_s\n")
        for r in results:
            for e in r["loss_history"]:
                f.write(
                    f"{r['label']},{r['budget_label']},{r['budget_minutes']},"
                    f"{r['n_params']},{r['config']['data_fraction']},"
                    f"{e['step']},{e['train_loss']:.6f},{e['val_loss']:.6f},"
                    f"{e['lr']:.8f},{e['elapsed_s']:.3f},"
                    f"{e['cumul_gflops']:.4f},{e['gflops_per_s']:.4f}\n"
                )
    return path


def save_results(results: list) -> str:
    path = os.path.join(LOGS_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    os.makedirs(LOGS_DIR, exist_ok=True)

    configs = build_configs(dry_run=dry_run)
    max_workers = min(len(configs), max(1, (os.cpu_count() or 4) - 2))

    print(
        f"{'[DRY RUN] ' if dry_run else ''}"
        f"Sweep: {len(configs)} configs  |  {max_workers} parallel workers"
    )
    print(
        f"Pi throughput estimate: {PI_GFLOPS_S} GFLOPs/s  "
        f"(from reference run on commit e06b38b, 2026-04-16)"
    )
    print(
        f"Eval overhead: {N_CHECKPOINTS} checkpoints × "
        f"{EVAL_ITERS} eval_iters per run"
    )
    print()

    # Print grid plan
    print(
        f"{'label':<22} {'budget':>6} {'n_layer':>7} {'params':>8} "
        f"{'data%':>6} {'max_iters':>9} {'tok_seen':>10} "
        f"{'est_train_min':>13} {'est_eval_min':>12} {'est_total_min':>13}"
    )
    print("-" * 100)

    budgets_seen = set()
    for cfg in configs:
        n_params = _param_count(cfg.n_layer)
        fps = 6 * n_params * BATCH_SIZE * CONTEXT_LENGTH
        batch_tokens = BATCH_SIZE * CONTEXT_LENGTH
        train_min = cfg.max_iters * fps / (PI_GFLOPS_S * 1e9) / 60
        eval_flops = N_CHECKPOINTS * 2 * EVAL_ITERS * 2 * n_params * batch_tokens
        eval_min = eval_flops / (PI_GFLOPS_S * 1e9) / 60
        tok_seen = cfg.max_iters * batch_tokens
        if cfg.budget_label not in budgets_seen:
            print()
            budgets_seen.add(cfg.budget_label)
        print(
            f"{cfg.label:<22} {cfg.budget_label:>6} {cfg.n_layer:>7} {n_params:>8,} "
            f"{int(cfg.data_fraction * 100):>5}% {cfg.max_iters:>9} "
            f"{tok_seen:>10,} {train_min:>13.1f} {eval_min:>12.1f} "
            f"{train_min + eval_min:>13.1f}"
        )

    print()
    print(f"Starting sweep...  (live logs in logs/sweep/)")
    sweep_start = time.time()

    results = sweep(configs, max_workers=max_workers)

    total_time = time.time() - sweep_start
    print(
        f"\nSweep complete: {len(results)}/{len(configs)} succeeded  "
        f"in {total_time / 60:.1f} min"
    )

    if results:
        print()
        print_summary_table(results)
        print_param_scaling_per_budget(results)
        print_budget_winners(results)

        json_path = save_results(results)
        csv_path = save_csv_loss_history(results)
        print(f"\nResults saved to: {json_path}")
        print(f"Loss history CSV: {csv_path}")
        print(f"\nRun  python plot_sweep.py  to generate plots.")