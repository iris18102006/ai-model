"""Train the tiny GPT on a text file.  python -m ai_model.train"""
import argparse
import time
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--out", default="checkpoints/ckpt.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(1337)

    text = Path(args.data).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"{len(text):,} chars, vocab {len(chars)}")

    cfg = GPTConfig(
        vocab_size=len(chars), block_size=args.block,
        n_layer=args.layers, n_head=args.heads, n_embd=args.embd,
    )
    model = GPT(cfg)
    print(f"params: {model.num_params() / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 10)

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch,))
        x = torch.stack([d[i:i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def estimate_loss(iters=20):
        model.eval()
        out = {}
        for split in ("train", "val"):
            ls = torch.zeros(iters)
            for k in range(iters):
                _, loss = model(*get_batch(split))
                ls[k] = loss.item()
            out[split] = ls.mean().item()
        model.train()
        return out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            l = estimate_loss()
            print(f"step {step:5d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time() - t0:.0f}s")
            torch.save(
                {"model": model.state_dict(), "config": cfg.to_dict(), "chars": chars},
                args.out,
            )
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    print(f"done. saved to {args.out}")


if __name__ == "__main__":
    main()
