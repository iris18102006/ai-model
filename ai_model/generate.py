"""Sample text from a trained checkpoint.  python -m ai_model.generate --prompt "ROMEO:" """
import argparse

import torch

from .model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    chars = ck["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])

    idx = torch.tensor([[stoi[c] for c in args.prompt if c in stoi]], dtype=torch.long)
    out = model.generate(idx, args.tokens, temperature=args.temp, top_k=args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
 
