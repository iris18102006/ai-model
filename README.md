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

## Browser demo

```bash
pip install numpy   # optional, silences a torch warning
python -m ai_model.export       # writes web/model.bin + web/model.json
cd web && python3 -m http.server 8000
```

Open http://localhost:8000. Inference is pure JS (`web/gpt.js`, no deps, KV cache).

Low on RAM/CPU? Try `--embd 64 --layers 2 --batch 16 --threads 2`.

## Roadmap

- [x] Model (attention, MLP, blocks, weight tying)
- [x] Training loop + checkpoints
- [x] Sampling script
- [x] Export weights for the browser
- [x] Browser demo
- [ ] Train on a custom dataset
