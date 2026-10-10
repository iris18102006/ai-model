"""Export a checkpoint for the browser demo.  python -m ai_model.export
Writes web/model.json (config, vocab, tensor layout) and web/model.bin (float32 weights)."""
import argparse
import json
from array import array
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--out", default="web")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest, buf, offset = {}, bytearray(), 0
    for name, t in sd.items():
        if name == "head.weight":  # tied to tok_emb.weight
            continue
        flat = t.detach().cpu().float().contiguous().flatten().tolist()
        buf += array("f", flat).tobytes()
        manifest[name] = {"shape": list(t.shape), "offset": offset, "length": len(flat)}
        offset += len(flat)

    (out / "model.bin").write_bytes(bytes(buf))
    (out / "model.json").write_text(
        json.dumps({"config": ck["config"], "chars": ck["chars"], "tensors": manifest})
    )
    print(f"wrote {out}/model.bin ({len(buf) / 1e6:.2f} MB) and {out}/model.json")


if __name__ == "__main__":
    main()
