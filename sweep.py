"""
Chinchilla-style IsoFLOPs sweep for the bike-powered LM.

For each compute budget (target wall-clock on a Raspberry Pi 3 at 1.72 GFLOPs/s
single-thread CPU), we train several model sizes. Training iterations fall out
from the budget: iters = budget_flops / (6·N·B·T). The winner per budget is
the compute-optimal model size; across budgets, diminishing returns tell us
how long the bike needs to spin.

    python sweep.py            # full sweep
    python sweep.py --dry-run  # 20 iters per config, validates pipeline

Results land in logs/sweep/{label}.log plus results.json for plotting.
"""

import argparse
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
)

# Raspberry Pi 3 single-thread throughput, measured on commit e06b38b.
PI_GFLOPS_S = 1.72

BUDGET_MINUTES = [30, 120, 240]  # 30m, 2h, 4h (Manhattan perimeter)

# (n_layer, n_embd) pairs — spans ~16k → ~306k params logarithmically.
MODEL_SIZES = [(1, 32), (1, 48), (2, 48), (2, 64), (4, 64), (6, 64)]

BATCH_SIZE = 64
CONTEXT_LENGTH = 64
N_HEAD = 4
LEARNING_RATE = 3e-3
LR_MIN = 1e-4
TRAIN_VAL_SPLIT = 0.9
EVAL_ITERS = 10
N_CHECKPOINTS = 10

DATA_PATH = os.path.join(_DIR, "dataset.txt")
LOGS_DIR = os.path.join(_DIR, "logs", "sweep")


@dataclass
class SweepConfig:
    label: str
    budget_minutes: int
    n_layer: int
    n_embd: int
    max_iters: int
    seed: int
    log_path: str


def run_config(cfg: SweepConfig) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)

    raw_data = get_data(DATA_PATH)
    vocab, vocab_size = create_vocabulary(raw_data)
    tokenizer = Tokenizer(vocab)
    train_data, val_data = get_train_val_data(
        raw_data, tokenizer, "cpu", TRAIN_VAL_SPLIT
    )

    model = GPT(
        GPTConfig(
            block_size=CONTEXT_LENGTH,
            vocab_size=vocab_size,
            n_layer=cfg.n_layer,
            n_head=N_HEAD,
            n_embd=cfg.n_embd,
            dropout=0.0,
            bias=True,
        )
    )
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = 6 * n_params * BATCH_SIZE * CONTEXT_LENGTH

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    eval_every = max(cfg.max_iters // N_CHECKPOINTS, 1)
    history = []
    start = time.time()

    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    with open(cfg.log_path, "w", buffering=1) as lf:
        lf.write(f"=== {cfg.label} ===\n")
        lf.write(f"budget_minutes  : {cfg.budget_minutes}\n")
        lf.write(f"n_layer,n_embd  : {cfg.n_layer}, {cfg.n_embd}\n")
        lf.write(f"n_params        : {n_params:,}\n")
        lf.write(f"max_iters       : {cfg.max_iters}\n")
        lf.write(f"flops_per_step  : {flops_per_step:,}\n")
        lf.write(f"tokens_seen     : {cfg.max_iters * BATCH_SIZE * CONTEXT_LENGTH:,}\n")
        lf.write(f"started_at      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.write(f"{'step':>6}  {'train':>8}  {'val':>8}  {'lr':>8}  {'elapsed':>8}\n")
        lf.write("-" * 50 + "\n")

        for step in range(cfg.max_iters):
            progress = step / max(cfg.max_iters - 1, 1)
            lr = LR_MIN + 0.5 * (LEARNING_RATE - LR_MIN) * (
                1 + math.cos(math.pi * progress)
            )
            for pg in opt.param_groups:
                pg["lr"] = lr

            x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % eval_every == 0 or step == cfg.max_iters - 1:
                tr = estimate_loss(
                    model, train_data, BATCH_SIZE, CONTEXT_LENGTH, EVAL_ITERS
                ).item()
                vl = estimate_loss(
                    model, val_data, BATCH_SIZE, CONTEXT_LENGTH, EVAL_ITERS
                ).item()
                elapsed = time.time() - start
                history.append(
                    {
                        "step": step,
                        "train_loss": tr,
                        "val_loss": vl,
                        "lr": lr,
                        "elapsed_s": elapsed,
                    }
                )
                lf.write(
                    f"{step:>6}  {tr:>8.4f}  {vl:>8.4f}  {lr:>8.5f}  {elapsed:>8.1f}\n"
                )

    elapsed = time.time() - start
    best_val = min(e["val_loss"] for e in history)
    total_flops = cfg.max_iters * flops_per_step
    return {
        "label": cfg.label,
        "budget_minutes": cfg.budget_minutes,
        "n_layer": cfg.n_layer,
        "n_embd": cfg.n_embd,
        "n_params": n_params,
        "max_iters": cfg.max_iters,
        "tokens_seen": cfg.max_iters * BATCH_SIZE * CONTEXT_LENGTH,
        "total_flops": total_flops,
        "best_val_loss": best_val,
        "final_val_loss": history[-1]["val_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "elapsed_s": elapsed,
        "history": history,
    }


def _derive_iters(budget_s, n_params):
    """Pick iters so Pi wall-clock ≈ budget_s, including eval overhead."""
    batch_tokens = BATCH_SIZE * CONTEXT_LENGTH
    budget_flops = budget_s * PI_GFLOPS_S * 1e9
    eval_flops = N_CHECKPOINTS * 2 * EVAL_ITERS * 2 * n_params * batch_tokens
    train_flops = max(budget_flops - eval_flops, 0)
    return max(50, int(train_flops / (6 * n_params * batch_tokens)))


def _count_params(n_layer, n_embd, vocab_size=35):
    per_block = 12 * n_embd * n_embd + 13 * n_embd
    return (vocab_size + CONTEXT_LENGTH) * n_embd + n_layer * per_block + 2 * n_embd


def build_configs(dry_run):
    configs = []
    cid = 0
    for bm in BUDGET_MINUTES:
        for n_layer, n_embd in MODEL_SIZES:
            n_params = _count_params(n_layer, n_embd)
            iters = 20 if dry_run else _derive_iters(bm * 60, n_params)
            label = f"b{bm}m_p{n_params // 1000}k"
            configs.append(
                SweepConfig(
                    label=label,
                    budget_minutes=bm,
                    n_layer=n_layer,
                    n_embd=n_embd,
                    max_iters=iters,
                    seed=42 + cid,
                    log_path=os.path.join(LOGS_DIR, f"{label}.log"),
                )
            )
            cid += 1
    return configs


def sweep(configs, max_workers):
    ctx = multiprocessing.get_context("fork")
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx
    ) as ex:
        futures = {ex.submit(run_config, c): c for c in configs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            c = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(
                    f"[{i:2d}/{len(configs)}] {c.label:<18}  "
                    f"params={r['n_params']:>6,}  iters={r['max_iters']:>5}  "
                    f"best_val={r['best_val_loss']:.4f}  {r['elapsed_s']/60:.1f}m"
                )
            except Exception as e:
                import traceback

                print(f"[{i:2d}/{len(configs)}] FAILED {c.label}: {e}")
                traceback.print_exc()
    results.sort(key=lambda r: (r["budget_minutes"], r["n_params"]))
    return results


def print_summary(results):
    print()
    print("=" * 72)
    print(
        f"{'budget':>7}  {'params':>8}  {'iters':>6}  {'tok_seen':>10}  "
        f"{'best_val':>9}  {'gap':>7}"
    )
    print("-" * 72)
    winners = {}
    for r in results:
        gap = r["final_val_loss"] - r["final_train_loss"]
        marker = ""
        bm = r["budget_minutes"]
        if bm not in winners or r["best_val_loss"] < winners[bm]["best_val_loss"]:
            winners[bm] = r
        print(
            f"{bm:>5}m   {r['n_params']:>8,}  {r['max_iters']:>6}  "
            f"{r['tokens_seen']:>10,}  {r['best_val_loss']:>9.4f}  {gap:>+7.4f}{marker}"
        )
    print("=" * 72)
    print("\nCompute-optimal winner per budget:")
    for bm in sorted(winners):
        w = winners[bm]
        print(
            f"  {bm:>3}m → {w['n_params']:>6,} params  "
            f"(n_layer={w['n_layer']}, n_embd={w['n_embd']})  "
            f"best_val={w['best_val_loss']:.4f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)
    configs = build_configs(args.dry_run)
    max_workers = min(len(configs), max(1, (os.cpu_count() or 4) - 2))

    print(
        f"{'[DRY] ' if args.dry_run else ''}"
        f"{len(configs)} configs × {max_workers} workers  |  "
        f"Pi={PI_GFLOPS_S} GFLOPs/s"
    )
    print(
        f"\n{'label':<16} {'budget':>6} {'params':>8} {'iters':>7} "
        f"{'tok_seen':>10} {'est_min':>8}"
    )
    for c in configs:
        n_params = _count_params(c.n_layer, c.n_embd)
        fps = 6 * n_params * BATCH_SIZE * CONTEXT_LENGTH
        est_min = c.max_iters * fps / (PI_GFLOPS_S * 1e9) / 60
        tok = c.max_iters * BATCH_SIZE * CONTEXT_LENGTH
        print(
            f"{c.label:<16} {c.budget_minutes:>5}m {n_params:>8,} "
            f"{c.max_iters:>7} {tok:>10,} {est_min:>8.1f}"
        )

    print(f"\nStarting sweep. Live logs in {os.path.relpath(LOGS_DIR)}/\n")
    t0 = time.time()
    results = sweep(configs, max_workers)
    print(f"\nDone in {(time.time() - t0)/60:.1f} min")

    if results:
        print_summary(results)
        out = os.path.join(LOGS_DIR, "results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: asdict(o))
        print(f"\nResults: {os.path.relpath(out)}")
        print("Plots:   python plot_sweep.py")
