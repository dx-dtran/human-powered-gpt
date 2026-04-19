import json
import os
import sys
import torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(_DIR, "weights", "bike_char.pth")
VOCAB_PATH = os.path.join(_DIR, "weights", "bike_char_vocab.json")
CONFIG_PATH = os.path.join(_DIR, "weights", "bike_char_config.json")

# Special token strings used in prompts/responses
_SPECIAL = {"[H]", "[A]", "[END]"}


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def _load_model():
    # Import model classes from train.py
    sys.path.insert(0, _DIR)
    from train import GPT, GPTConfig

    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    cfg_kwargs = _load_config()

    print(
        f"  weights   : {WEIGHTS_PATH}\n"
        f"  vocab     : {cfg_kwargs['vocab_size']} chars\n"
        f"  block_size: {cfg_kwargs['block_size']}\n"
        f"  layers    : {cfg_kwargs['n_layer']}  heads: {cfg_kwargs['n_head']}  d_embd: {cfg_kwargs['n_embd']}\n"
    )

    model = GPT(GPTConfig(**cfg_kwargs))
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg_kwargs["block_size"]


def _stream_response(
    model, context, block_size, id_to_char, max_new=300, temperature=0.8
):
    """Yield decoded characters one by one, suppressing special tokens."""
    # All special tokens start with '[', so we only buffer when we see '['.
    # Outside of '[...', every char is immediately safe to yield.
    pending = ""  # non-empty only while inside a potential special token

    with torch.no_grad():
        for _ in range(max_new):
            ctx = context[:, -block_size:]
            scores, _ = model(ctx)
            logits = scores[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ch = id_to_char.get(next_id, "?")
            context = torch.cat([context, torch.tensor([[next_id]])], dim=1)

            if not pending and ch != "[":
                yield ch
                continue

            pending += ch

            # Is pending a complete special token?
            if pending in _SPECIAL:
                if pending == "[END]":
                    return
                pending = ""  # suppress [H] / [A]
                continue

            # Is pending still a valid prefix of any special token?
            if any(tok.startswith(pending) for tok in _SPECIAL):
                continue  # keep buffering

            # Not a prefix anymore — flush pending as literal text then reset
            for c in pending:
                yield c
            pending = ""

    # Flush any leftover pending chars
    for c in pending:
        yield c


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading EnerGPT...\n")
    model, block_size = _load_model()
    model = model.to(device)

    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_char = {i: ch for i, ch in enumerate(vocab)}

    def encode(text):
        return [char_to_id.get(ch, 0) for ch in text]

    print('Type your message and press Enter. Type "quit" to exit.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("bye")
            break

        prompt = f"[H] {user_input.lower()} [A] "
        prompt_ids = encode(prompt)
        if len(prompt_ids) >= block_size:
            prompt_ids = prompt_ids[-(block_size - 1) :]

        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        print("EnerGPT: ", end="", flush=True)
        for ch in _stream_response(model, context, block_size, id_to_char):
            print(ch, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
