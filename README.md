# ai-model

A tiny GPT built from scratch in PyTorch. Char-level, CPU friendly, ~0.8M params. Browser demo coming later.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
python data/get_data.py
python -m ai_model.train --steps 3000
python -m ai_model.generate --prompt "ROMEO:"
```

Low on RAM/CPU? Try `--embd 64 --layers 2 --batch 16 --threads 2`.

## Roadmap

- [x] Model (attention, MLP, blocks, weight tying)
- [x] Training loop + checkpoints
- [x] Sampling script
- [ ] Export weights for the browser
- [ ] Browser demo
- [ ] Train on a custom dataset
