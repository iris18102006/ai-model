// Tiny GPT inference in plain JS (no deps). Mirrors ai_model/model.py with a KV cache.
class TinyGPT {
  constructor(meta, weights) {
    this.cfg = meta.config;
    this.chars = meta.chars;
    this.stoi = new Map(this.chars.map((c, i) => [c, i]));
    this.w = {};
    for (const [name, t] of Object.entries(meta.tensors)) {
      this.w[name] = new Float32Array(weights, t.offset * 4, t.length);
    }
    const { n_layer, n_embd, block_size } = this.cfg;
    this.k = []; this.v = [];
    for (let l = 0; l < n_layer; l++) {
      this.k.push(new Float32Array(block_size * n_embd));
      this.v.push(new Float32Array(block_size * n_embd));
    }
    this.len = 0;
  }

  reset() { this.len = 0; }

  static erf(x) { // Abramowitz-Stegun 7.1.26
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  static layerNorm(x, g, b, out) {
    const n = x.length; let m = 0, v = 0;
    for (let i = 0; i < n; i++) m += x[i]; m /= n;
    for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv * g[i] + b[i];
  }

  static linear(W, b, x, rows, cols, out) { // W: [rows, cols]
    for (let i = 0; i < rows; i++) {
      let s = b ? b[i] : 0; const o = i * cols;
      for (let j = 0; j < cols; j++) s += W[o + j] * x[j];
      out[i] = s;
    }
  }

  // Feed one token at position this.len, return logits (Float32Array[vocab]).
  step(tokenId) {
    const { n_layer, n_head, n_embd: C, block_size, vocab_size } = this.cfg;
    const hd = C / n_head, pos = this.len, w = this.w;
    if (pos >= block_size) throw new Error("context full");
    const x = new Float32Array(C);
    for (let i = 0; i < C; i++) x[i] = w["tok_emb.weight"][tokenId * C + i] + w["pos_emb.weight"][pos * C + i];

    const h = new Float32Array(C), qkv = new Float32Array(3 * C), y = new Float32Array(C);
    const tmp = new Float32Array(C), ff = new Float32Array(4 * C), ffo = new Float32Array(C);
    const att = new Float32Array(block_size);

    for (let l = 0; l < n_layer; l++) {
      const p = `blocks.${l}.`;
      TinyGPT.layerNorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"], h);
      TinyGPT.linear(w[p + "attn.qkv.weight"], w[p + "attn.qkv.bias"], h, 3 * C, C, qkv);
      this.k[l].set(qkv.subarray(C, 2 * C), pos * C);
      this.v[l].set(qkv.subarray(2 * C, 3 * C), pos * C);
      const K = this.k[l], V = this.v[l], scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < n_head; hh++) {
        const ho = hh * hd; let mx = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0; for (let d = 0; d < hd; d++) s += qkv[ho + d] * K[t * C + ho + d];
          att[t] = s * scale; if (att[t] > mx) mx = att[t];
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { att[t] = Math.exp(att[t] - mx); sum += att[t]; }
        for (let d = 0; d < hd; d++) {
          let s = 0; for (let t = 0; t <= pos; t++) s += att[t] * V[t * C + ho + d];
          y[ho + d] = s / sum;
        }
      }
      TinyGPT.linear(w[p + "attn.proj.weight"], w[p + "attn.proj.bias"], y, C, C, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];

      TinyGPT.layerNorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"], h);
      TinyGPT.linear(w[p + "mlp.fc.weight"], w[p + "mlp.fc.bias"], h, 4 * C, C, ff);
      for (let i = 0; i < 4 * C; i++) ff[i] = 0.5 * ff[i] * (1 + TinyGPT.erf(ff[i] / Math.SQRT2));
      TinyGPT.linear(w[p + "mlp.proj.weight"], w[p + "mlp.proj.bias"], ff, C, 4 * C, ffo);
      for (let i = 0; i < C; i++) x[i] += ffo[i];
    }
    TinyGPT.layerNorm(x, w["ln_f.weight"], w["ln_f.bias"], h);
    const logits = new Float32Array(vocab_size), E = w["tok_emb.weight"]; // tied head
    for (let i = 0; i < vocab_size; i++) {
      let s = 0; for (let j = 0; j < C; j++) s += E[i * C + j] * h[j];
      logits[i] = s;
    }
    this.len++;
    return logits;
  }

  // Feed many tokens, return logits of the last one.
  feed(ids) { let lg; for (const id of ids) lg = this.step(id); return lg; }

  static sample(logits, temperature, topK) {
    const n = logits.length, l = Array.from(logits, v => v / Math.max(temperature, 1e-5));
    if (topK > 0 && topK < n) {
      const th = [...l].sort((a, b) => b - a)[topK - 1];
      for (let i = 0; i < n; i++) if (l[i] < th) l[i] = -Infinity;
    }
    const mx = Math.max(...l); let sum = 0;
    const p = l.map(v => { const e = Math.exp(v - mx); sum += e; return e; });
    let r = Math.random() * sum;
    for (let i = 0; i < n; i++) { r -= p[i]; if (r <= 0) return i; }
    return n - 1;
  }

  encode(text) { return [...text].filter(c => this.stoi.has(c)).map(c => this.stoi.get(c)); }

  // Generator: yields one char at a time. Slides the window when context fills up.
  *generate(prompt, maxNew, temperature, topK) {
    const B = this.cfg.block_size;
    let ids = this.encode(prompt); if (!ids.length) ids = this.encode("\n");
    this.reset();
    let logits = this.feed(ids.slice(-B));
    for (let i = 0; i < maxNew; i++) {
      const nxt = TinyGPT.sample(logits, temperature, topK);
      ids.push(nxt);
      yield this.chars[nxt];
      if (this.len >= B) { // re-prefill the last half so positions stay in range
        this.reset();
        logits = this.feed(ids.slice(-(B >> 1)));
      } else logits = this.step(nxt);
    }
  }
}
if (typeof module !== "undefined") module.exports = { TinyGPT };
