"""
Parallel hyperparameter sweep for scaling law analysis.

Sweeps 4 model sizes x 3 data fractions = 12 configs, all targeting the same
FLOP budget (~8 TFLOPs), so results are directly comparable at equal compute.

Usage:
    python sweep.py            # full sweep
    python sweep.py --dry-run  # 10 iters per config (~30s), validates pipeline
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

# ── constants ────────────────────────────────────────────────────────────────
TARGET_FLOPS = 8e12        # ~56-58 min per config on a Raspberry Pi
BATCH_SIZE = 64
CONTEXT_LENGTH = 64
LEARNING_RATE = 3e-3
LR_MIN = 1e-4
EVAL_ITERS = 20            # batches averaged per loss estimate
TRAIN_VAL_SPLIT = 0.9

DATA_PATH = os.path.join(_DIR, "dataset.txt")
LOGS_DIR = os.path.join(_DIR, "logs", "sweep")


# ── config dataclass ─────────────────────────────────────────────────────────
@dataclass
class SweepConfig:
    label: str
    n_layer: int
    data_fraction: float
    max_iters: int
    n_embd: int = 64
    n_head: int = 4
    batch_size: int = BATCH_SIZE
    context_length: int = CONTEXT_LENGTH
    seed: int = 42
    log_path: str = ""


# ── worker ───────────────────────────────────────────────────────────────────
def run_config(cfg: SweepConfig) -> dict:
    """Train one config. Module-level so ProcessPoolExecutor can pickle it."""
    torch.set_num_threads(1)   # prevent per-process thread oversubscription
    torch.manual_seed(cfg.seed)
    device = "cpu"

    raw_data = get_data(DATA_PATH)
    vocab, vocab_size = create_vocabulary(raw_data)
    tokenizer = Tokenizer(vocab)
    train_data, val_data = get_train_val_data(raw_data, tokenizer, device, TRAIN_VAL_SPLIT)

    # Slice train data; val is always the full split for fair comparison
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

    eval_interval = max(cfg.max_iters // 20, 50)
    loss_history = []
    cumulative_flops = 0.0
    window_flops = 0.0
    window_start = time.time()
    start = time.time()

    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    with open(cfg.log_path, "w", buffering=1) as lf:
        lf.write(f"=== run: {cfg.label} ===\n")
        lf.write(f"label           : {cfg.label}\n")
        lf.write(f"params          : {n_params:,}\n")
        lf.write(f"n_embd          : {cfg.n_embd}\n")
        lf.write(f"n_layer         : {cfg.n_layer}\n")
        lf.write(f"n_head          : {cfg.n_head}\n")
        lf.write(f"data_fraction   : {cfg.data_fraction}\n")
        lf.write(f"train_tokens    : {n_train:,}\n")
        lf.write(f"val_tokens      : {len(val_data):,}\n")
        lf.write(f"max_iters       : {cfg.max_iters}\n")
        lf.write(f"eval_interval   : {eval_interval}\n")
        lf.write(f"batch_size      : {cfg.batch_size}\n")
        lf.write(f"context_length  : {cfg.context_length}\n")
        lf.write(f"seed            : {cfg.seed}\n")
        lf.write(f"flops_per_step  : {flops_per_step:,}\n")
        lf.write(f"target_flops    : {int(TARGET_FLOPS):,}\n")
        lf.write(f"device          : {device}\n")
        lf.write(f"started_at      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write("\n")

        # Column header
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

    return {
        "label": cfg.label,
        "config": asdict(cfg),
        "n_params": n_params,
        "vocab_size": vocab_size,
        "n_train_tokens": n_train,
        "n_val_tokens": len(val_data),
        "flops_per_step": flops_per_step,
        "loss_history": loss_history,
        "final_train_loss": loss_history[-1]["train_loss"] if loss_history else None,
        "final_val_loss": loss_history[-1]["val_loss"] if loss_history else None,
        "best_val_loss": min(e["val_loss"] for e in loss_history) if loss_history else None,
        "total_flops": cumulative_flops,
        "elapsed_s": total_elapsed,
    }


# ── sweep orchestration ───────────────────────────────────────────────────────
def _param_count(n_layer, n_embd=64, vocab_size=35, block_size=64):
    """Compute exact parameter count matching the GPT model structure in train.py."""
    per_block = (
        3 * n_embd * n_embd + 3 * n_embd   # c_attn weight + bias
        + n_embd * n_embd + n_embd          # c_proj weight + bias
        + 4 * n_embd * n_embd + 4 * n_embd # mlp c_fc weight + bias
        + 4 * n_embd * n_embd + n_embd      # mlp c_proj weight + bias
        + 2 * n_embd                        # ln_1 weight + bias
        + 2 * n_embd                        # ln_2 weight + bias
    )
    return vocab_size * n_embd + block_size * n_embd + n_layer * per_block + 2 * n_embd


def build_configs(dry_run: bool = False) -> list:
    """Build the 12-config grid: 4 n_layer values x 3 data fractions."""
    LAYERS = [1, 2, 4, 6]
    FRACS = [0.25, 0.5, 1.0]

    configs = []
    for idx, n_layer in enumerate(LAYERS):
        n_params = _param_count(n_layer)
        flops_per_step = 6 * n_params * BATCH_SIZE * CONTEXT_LENGTH
        for jdx, frac in enumerate(FRACS):
            config_id = idx * len(FRACS) + jdx
            max_iters = 10 if dry_run else max(50, int(TARGET_FLOPS / flops_per_step))
            params_k = n_params // 1000
            label = f"{params_k}k_d{int(frac * 100)}"
            log_path = os.path.join(LOGS_DIR, f"{label}.log")
            configs.append(
                SweepConfig(
                    label=label,
                    n_layer=n_layer,
                    data_fraction=frac,
                    max_iters=max_iters,
                    seed=42 + config_id,
                    log_path=log_path,
                )
            )
    return configs


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
                    f"[{completed:2d}/{total}]  {cfg.label:<14}  "
                    f"train={result['final_train_loss']:.4f}  "
                    f"val={result['final_val_loss']:.4f}  "
                    f"{result['total_flops'] / 1e9:.1f} GF  "
                    f"{result['elapsed_s'] / 60:.1f} min  "
                    f"log -> {os.path.relpath(cfg.log_path)}"
                )
            except Exception as exc:
                import traceback
                print(f"[{completed:2d}/{total}]  FAILED {cfg.label}: {exc}")
                traceback.print_exc()

    results.sort(key=lambda r: (r["n_params"], r["config"]["data_fraction"]))
    return results


# ── report printing ───────────────────────────────────────────────────────────
def print_summary_table(results: list) -> None:
    cols = (
        f"{'label':<16} {'params':>8} {'n_layer':>7} {'data%':>6} "
        f"{'iters':>6} {'train_loss':>10} {'val_loss':>9} "
        f"{'best_val':>9} {'GFLOPs':>8} {'min':>7}"
    )
    bar = "=" * len(cols)
    print(bar)
    print("SWEEP SUMMARY")
    print(bar)
    print(cols)
    print("-" * len(cols))
    for r in results:
        cfg = r["config"]
        print(
            f"{r['label']:<16} {r['n_params']:>8,} {cfg['n_layer']:>7} "
            f"{int(cfg['data_fraction'] * 100):>5}% "
            f"{cfg['max_iters']:>6} {r['final_train_loss']:>10.4f} "
            f"{r['final_val_loss']:>9.4f} {r['best_val_loss']:>9.4f} "
            f"{r['total_flops'] / 1e9:>8.1f} {r['elapsed_s'] / 60:>7.2f}"
        )
    print(bar)


def print_scaling_analysis(results: list) -> None:
    print()
    print("=" * 62)
    print("SCALING ANALYSIS")
    print("=" * 62)

    full = [r for r in results if abs(r["config"]["data_fraction"] - 1.0) < 0.01]
    if full:
        print("\n--- Parameter Scaling (data=100%, isoFLOP ~8 TF) ---")
        for r in sorted(full, key=lambda x: x["n_params"]):
            print(
                f"  {r['n_params']:>8,} params  n_layer={r['config']['n_layer']}  "
                f"best_val={r['best_val_loss']:.4f}  final_val={r['final_val_loss']:.4f}"
            )

    for nl in sorted(set(r["config"]["n_layer"] for r in results)):
        group = sorted(
            [r for r in results if r["config"]["n_layer"] == nl],
            key=lambda x: x["config"]["data_fraction"],
        )
        if not group:
            continue
        n_params = group[0]["n_params"]
        print(f"\n--- Data Scaling ({n_params:,} params, n_layer={nl}) ---")
        for r in group:
            frac = r["config"]["data_fraction"]
            print(
                f"  data={int(frac * 100):3d}%  ({r['n_train_tokens']:,} tokens)  "
                f"best_val={r['best_val_loss']:.4f}  final_val={r['final_val_loss']:.4f}"
            )


def print_csv_loss_history(results: list) -> None:
    print()
    print("# ── LOSS HISTORY CSV ────────────────────────────────────────────────────────")
    print("# Copy everything below this line into a .csv file for external charting.")
    print("# label,step,train_loss,val_loss,lr,elapsed_s,cumul_gflops,gflops_per_s")
    for r in results:
        for e in r["loss_history"]:
            print(
                f"{r['label']},"
                f"{e['step']},"
                f"{e['train_loss']:.6f},"
                f"{e['val_loss']:.6f},"
                f"{e['lr']:.8f},"
                f"{e['elapsed_s']:.3f},"
                f"{e['cumul_gflops']:.4f},"
                f"{e['gflops_per_s']:.4f}"
            )


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
        f"FLOP target: {TARGET_FLOPS / 1e12:.0f} TF/run  |  "
        f"batch={BATCH_SIZE}  context={CONTEXT_LENGTH}"
    )
    print()
    print(
        f"{'label':<16} {'n_layer':>7} {'~params':>9} {'data%':>6} "
        f"{'max_iters':>9} {'est_pi_min':>10}"
    )
    print("-" * 62)
    pi_gflops_s = 2.23  # estimated from 39-min Pi reference run
    for cfg in configs:
        n_params = _param_count(cfg.n_layer)
        fps = 6 * n_params * BATCH_SIZE * CONTEXT_LENGTH
        est_pi_s = cfg.max_iters * fps / (pi_gflops_s * 1e9)
        print(
            f"{cfg.label:<16} {cfg.n_layer:>7} {n_params:>9,} "
            f"{int(cfg.data_fraction * 100):>5}% "
            f"{cfg.max_iters:>9} {est_pi_s / 60:>10.1f}"
        )
    print()
    print("Starting sweep...  (live logs in logs/sweep/)")
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
        print_scaling_analysis(results)
        print_csv_loss_history(results)
        json_path = save_results(results)
        print(f"\nFull results (with complete loss histories) saved to: {json_path}")
