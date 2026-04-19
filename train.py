import json
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "a")

    def log(self, msg=""):
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def create_vocabulary(data):
    vocab = sorted(set(data))
    return vocab, len(vocab)


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._atoi = {ch: i for i, ch in enumerate(vocab)}
        self._itoa = {i: ch for i, ch in enumerate(vocab)}

    def encode(self, text):
        return [self._atoi[ch] for ch in text]

    def decode(self, indices):
        return "".join(self._itoa[i] for i in indices)


from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = c // self.n_head
        k = k.view(b, t, self.n_head, head_size).transpose(1, 2)
        q = q.view(b, t, self.n_head, head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        att = att.masked_fill(
            torch.tril(torch.ones(t, t, device=x.device)).view(1, 1, t, t) == 0,
            float("-inf"),
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.resid_dropout(self.c_proj(out))
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)

    def forward(self, indices, targets=None):
        b, t = indices.shape
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=indices.device)
        x = self.drop(self.wte(indices) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    @classmethod
    def from_pretrained(cls, weights_path, config: GPTConfig = None):
        if config is None:
            config = GPTConfig()
        model = cls(config)
        state_dict = torch.load(weights_path, map_location="cpu")
        # OpenAI weights live under a "transformer." prefix; strip it
        # lm_head.weight is tied to wte.weight — skip it
        _conv1d_keys = {
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        }
        new_sd = {}
        for k, v in state_dict.items():
            if k == "lm_head.weight":
                continue
            nk = k.removeprefix("transformer.")
            if any(nk.endswith(ck) for ck in _conv1d_keys):
                v = v.T
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=True)
        return model


def get_data(path):
    with open(path, "r") as f:
        return f.read()


def get_train_val_data(data, tokenizer, device, train_val_split=0.9):
    encoded_data = tokenizer.encode(data)
    data_tensor = torch.tensor(encoded_data, device=device)
    n = int(len(data_tensor) * train_val_split)
    return data_tensor[:n], data_tensor[n:]


def get_batch(data, batch_size, context_length):
    x, y = [], []
    for i in range(batch_size):
        index = torch.randint(0, len(data) - context_length - 1, (1,))
        x.append(data[index : index + context_length])
        y.append(data[index + 1 : index + context_length + 1])
    x, y = torch.stack(x), torch.stack(y)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for iteration in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length)
        _, loss = model(x, y)
        losses[iteration] = loss
    model.train()
    return losses.mean()


def _fmt_flops(flops):
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    return f"{flops / 1e9:.1f} GFLOPs"


def _fmt_throughput(flops_per_s):
    g = flops_per_s / 1e9
    if g >= 1.0:
        return f"{g:.2f} GFLOPs/s"
    return f"{g * 1000:.0f} MFLOPs/s"


def _fmt_eta(seconds):
    s = int(seconds)
    if s <= 0:
        return "done"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


_PI_GFLOPS_S = 1.72   # Raspberry Pi 3 single-thread throughput (matches sweep.py)
_N_CHECKPOINTS = 10   # eval checkpoints during training (matches sweep.py)
_BUDGET_MINUTES = 30  # target wall-clock on Pi


def _derive_train_iters(n_params, batch_size, context_length, budget_minutes=_BUDGET_MINUTES):
    """Derive training iterations from a Pi wall-clock budget, accounting for eval overhead.

    Uses the same formula as sweep.py so training dynamics are identical to a sweep run.
    The eval overhead subtracted here uses sweep's EVAL_ITERS=10; the main loop may run
    more eval iters for better estimates without affecting the training step count.
    """
    batch_tokens = batch_size * context_length
    budget_flops = budget_minutes * 60 * _PI_GFLOPS_S * 1e9
    # Account for eval overhead using sweep's 10-iter evals (not our 100-iter monitoring evals)
    eval_flops = _N_CHECKPOINTS * 2 * 10 * 2 * n_params * batch_tokens
    train_flops = max(budget_flops - eval_flops, 0)
    return max(50, int(train_flops / (6 * n_params * batch_tokens)))


def train():
    context_length = 64
    d_embed = 48
    n_head = 4  # match sweep.py N_HEAD
    n_layer = 2

    batch_size = 64
    eval_iters = 100
    learning_rate = 3e-3
    lr_min = 1e-4
    train_val_split = 0.9

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = os.path.join(_DIR, "dataset.txt")
    weights_path = os.path.join(_DIR, "weights", "bike_char.pth")
    vocab_path = os.path.join(_DIR, "weights", "bike_char_vocab.json")
    log_path = os.path.join(_DIR, "logs", "bike_char.log")

    os.makedirs(os.path.join(_DIR, "weights"), exist_ok=True)

    log = Logger(log_path)
    log.log("=== train_bike_char  {} ===".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    log.log(f"loading {data_path}...")

    raw_data = get_data(data_path)
    vocab, vocab_size = create_vocabulary(raw_data)
    tokenizer = Tokenizer(vocab)

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    train_data, val_data = get_train_val_data(
        raw_data, tokenizer, device, train_val_split
    )

    mygpt = GPT(
        GPTConfig(
            block_size=context_length,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=d_embed,
            dropout=0.0,  # match sweep.py (dropout=0.1 was hurting loss)
            bias=True,
        )
    ).to(device)

    n_params = sum(p.numel() for p in mygpt.parameters())
    flops_per_step = 6 * n_params * batch_size * context_length

    # Derive max_iters from the Pi budget using the same formula as sweep.py so
    # training dynamics are identical to the corresponding sweep run.
    max_iters = _derive_train_iters(n_params, batch_size, context_length)
    # Space 10 eval checkpoints evenly across training (same cadence as sweep)
    eval_interval = max(max_iters // _N_CHECKPOINTS, 1)

    log.log(f"vocab size    : {vocab_size} chars")
    log.log(f"train tokens  : {len(train_data):,}   val tokens: {len(val_data):,}")
    log.log(f"parameters    : {n_params:,}")
    log.log(
        f"context length: {context_length}   d_embed: {d_embed}   layers: {n_layer}   heads: {n_head}"
    )
    log.log(
        f"batch size    : {batch_size}   max iters: {max_iters}   lr: {learning_rate} -> {lr_min}"
    )
    log.log(f"device        : {device}")
    log.log(f"flops/step    : {_fmt_flops(flops_per_step)}")
    log.log("")

    optimizer = torch.optim.AdamW(mygpt.parameters(), lr=learning_rate)

    start = time.time()
    best_val_loss = float("inf")
    cumulative_flops = 0
    window_start = start
    window_flops = 0

    for step in range(max_iters):
        progress = step / max(max_iters - 1, 1)
        lr = lr_min + 0.5 * (learning_rate - lr_min) * (
            1 + math.cos(math.pi * progress)
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, batch_size, context_length)
        _, loss = mygpt(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumulative_flops += flops_per_step
        window_flops += flops_per_step

        if step % eval_interval == 0 or step == max_iters - 1:
            eval_start = time.time()
            window_elapsed = max(eval_start - window_start, 1e-6)
            flops_per_s = window_flops / window_elapsed

            train_loss = estimate_loss(
                mygpt, train_data, batch_size, context_length, eval_iters
            )
            val_loss = estimate_loss(
                mygpt, val_data, batch_size, context_length, eval_iters
            )
            saved = val_loss < best_val_loss
            if saved:
                best_val_loss = val_loss
                torch.save(mygpt.state_dict(), weights_path)

            sample_chars = []
            with torch.no_grad():
                mygpt.eval()
                sample_ctx = torch.tensor([[0]], dtype=torch.long, device=device)
                for _ in range(120):
                    c = sample_ctx[:, -mygpt.config.block_size :]
                    scores, _ = mygpt(c)
                    probs = torch.softmax(scores[0, -1], dim=-1)
                    nxt = torch.multinomial(probs, 1).item()
                    sample_chars.append(tokenizer.decode([nxt]))
                    sample_ctx = torch.cat(
                        [sample_ctx, torch.tensor([[nxt]], device=device)], dim=1
                    )
                mygpt.train()

            elapsed = time.time() - start
            steps_done = step + 1
            eta_s = (max_iters - steps_done) * (elapsed / steps_done)

            tag = " [saved]" if saved else ""
            log.log(
                "step {}/{} | {} | lr {:.5f} | train {:.4f} | val {:.4f} | {:.1f}s | {} | {} | eta {}{}".format(
                    step,
                    max_iters,
                    time.strftime("%H:%M:%S"),
                    lr,
                    train_loss,
                    val_loss,
                    elapsed,
                    _fmt_flops(cumulative_flops),
                    _fmt_throughput(flops_per_s),
                    _fmt_eta(eta_s),
                    tag,
                )
            )
            log.log("")
            log.log("sample text:")
            log.log("".join(sample_chars))
            log.log("")

            window_start = time.time()
            window_flops = 0

    log.log("total time : {:.1f}s".format(time.time() - start))
    log.log("best val   : {:.4f}".format(best_val_loss))
    log.log("weights    : {}".format(weights_path))
    log.close()


def chat(
    context_length=64,
    d_embed=64,
    n_head=4,
    n_layer=2,
    max_new=200,
    temperature=0.8,
):
    weights_path = os.path.join(_DIR, "weights", "bike_char.pth")
    vocab_path = os.path.join(_DIR, "weights", "bike_char_vocab.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    id_to_char = {i: ch for i, ch in enumerate(vocab)}
    char_to_id = {ch: i for i, ch in enumerate(vocab)}

    def encode(text):
        return [char_to_id.get(ch, 0) for ch in text]

    def decode(ids):
        return "".join(id_to_char.get(i, "?") for i in ids)

    model = GPT(
        GPTConfig(
            block_size=context_length,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=d_embed,
            dropout=0.0,
            bias=True,
        )
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print("bike model loaded. type your question or 'quit' to exit.\n")

    with torch.no_grad():
        while True:
            user_input = input("you: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            user_input = user_input.lower()
            prompt = f"[H] {user_input} [A]"
            prompt_ids = encode(prompt)

            if len(prompt_ids) >= context_length:
                prompt_ids = prompt_ids[-(context_length - 1) :]

            context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output_chars = []

            for _ in range(max_new):
                ctx = context[:, -context_length:]
                scores, _ = model(ctx)
                logits = scores[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                output_chars.append(id_to_char.get(next_id, "?"))
                context = torch.cat(
                    [context, torch.tensor([[next_id]], device=device)], dim=1
                )
                response_so_far = "".join(output_chars)
                if "[END]" in response_so_far:
                    output_chars = list(response_so_far.split("[END]")[0])
                    break

            print(f"bike: {''.join(output_chars).strip()}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        train()
