/* Detector core: shared by the page and the Node test. No DOM access here. */
(function (root) {
  "use strict";

  function b64bytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function median(arr) {
    const a = Float64Array.from(arr).sort();
    const n = a.length, h = n >> 1;
    return n % 2 ? a[h] : (a[h - 1] + a[h]) / 2;
  }

  // SExtractor-style background: per-mesh medians, 3x3 median smoothing, bilinear interpolation.
  function meshBackground(img, N, mesh) {
    const nb = Math.ceil(N / mesh);
    const grid = new Float64Array(nb * nb);
    for (let i = 0; i < nb; i++) for (let j = 0; j < nb; j++) {
      const vals = [];
      for (let r = i * mesh; r < Math.min((i + 1) * mesh, N); r++)
        for (let c = j * mesh; c < Math.min((j + 1) * mesh, N); c++) vals.push(img[r * N + c]);
      grid[i * nb + j] = median(vals);
    }
    const sm = new Float64Array(nb * nb);
    for (let i = 0; i < nb; i++) for (let j = 0; j < nb; j++) {
      const vals = [];
      for (let a = Math.max(i - 1, 0); a < Math.min(i + 2, nb); a++)
        for (let b = Math.max(j - 1, 0); b < Math.min(j + 2, nb); b++) vals.push(grid[a * nb + b]);
      sm[i * nb + j] = median(vals);
    }
    const k0 = new Int32Array(N), k1 = new Int32Array(N), f = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const x = (i + 0.5) / mesh - 0.5;
      k0[i] = Math.min(Math.max(Math.floor(x), 0), nb - 1);
      k1[i] = Math.min(k0[i] + 1, nb - 1);
      f[i] = Math.min(Math.max(x - k0[i], 0), 1);
    }
    const top = new Float64Array(N * nb);
    for (let i = 0; i < N; i++) for (let j = 0; j < nb; j++)
      top[i * nb + j] = sm[k0[i] * nb + j] * (1 - f[i]) + sm[k1[i] * nb + j] * f[i];
    const bg = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let c = 0; c < N; c++)
      bg[i * N + c] = top[i * nb + k0[c]] * (1 - f[c]) + top[i * nb + k1[c]] * f[c];
    const resid = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) resid[i] = img[i] - bg[i];
    const m = median(resid);
    for (let i = 0; i < N * N; i++) resid[i] = Math.abs(resid[i] - m);
    return { bg, sigma: median(resid) * 1.4826 };
  }

  // pixel equals the maximum of its (edge-clipped) 5x5 neighbourhood
  function localMax(img, N, rad) {
    const tmp = new Float64Array(N * N), mx = new Float64Array(N * N), out = new Uint8Array(N * N);
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      let m = -Infinity;
      for (let k = Math.max(c - rad, 0); k <= Math.min(c + rad, N - 1); k++) m = Math.max(m, img[r * N + k]);
      tmp[r * N + c] = m;
    }
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      let m = -Infinity;
      for (let k = Math.max(r - rad, 0); k <= Math.min(r + rad, N - 1); k++) m = Math.max(m, tmp[k * N + c]);
      mx[r * N + c] = m;
    }
    for (let i = 0; i < N * N; i++) out[i] = img[i] === mx[i] ? 1 : 0;
    return out;
  }

  function build(data, model) {
    const N = data.N;
    const u16 = new Uint16Array(b64bytes(data.dirty).buffer);
    const img = new Float64Array(N * N);
    const step = (data.max - data.min) / 65535;
    for (let i = 0; i < N * N; i++) img[i] = data.min + u16[i] * step;
    const esBits = b64bytes(data.es);
    const es = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) es[i] = (esBits[i >> 3] >> (i & 7)) & 1;
    const psNear = new Int16Array(N * N).fill(-1);   // index of a true point source within 2 px
    data.ps.forEach(([r, c], k) => {
      for (let a = Math.max(r - 2, 0); a <= Math.min(r + 2, N - 1); a++)
        for (let b = Math.max(c - 2, 0); b <= Math.min(c + 2, N - 1); b++)
          if (psNear[a * N + b] < 0) psNear[a * N + b] = k;
    });
    const { bg, sigma } = meshBackground(img, N, 64);
    const ismax = localMax(img, N, 2);
    const W = Float64Array.from(model.W);

    function classify(p, q) {
      let d = model.B;
      for (let jr = 0; jr < 51; jr++) {
        const r = p - 26 + jr;
        if (r < 0 || r >= N) continue;
        for (let jc = 0; jc < 51; jc++) {
          const c = q - 26 + jc;
          if (c < 0 || c >= N) continue;
          d += W[jr * 51 + jc] * img[r * N + c];
        }
      }
      let pr = 1 / (1 + Math.exp(model.A * d + model.Bp));
      if (model.pIsComplement) pr = 1 - pr;
      return { d, pPS: pr };
    }

    function truth(p, q) {
      const k = psNear[p * N + q];
      if (k >= 0) return { kind: "ps", flux: data.ps[k][2] };
      return { kind: es[p * N + q] ? "es" : "bg" };
    }

    function describe(p, q, tile) {
      const { d, pPS } = classify(p, q);
      const isPS = pPS >= 0.5;
      return {
        p, q, tile, d, pPS, isPS, conf: isPS ? pPS : 1 - pPS,
        prom: (img[p * N + q] - bg[p * N + q]) / sigma, truth: truth(p, q)
      };
    }

    function scan(T, thr) {
      const out = [];
      for (let r0 = 0; r0 < N; r0 += T) for (let c0 = 0; c0 < N; c0 += T) {
        let best = -Infinity, bp = -1, bq = -1;
        for (let r = r0; r < Math.min(r0 + T, N); r++) for (let c = c0; c < Math.min(c0 + T, N); c++) {
          const i = r * N + c;
          if (ismax[i] && img[i] > best) { best = img[i]; bp = r; bq = c; }
        }
        if (bp < 0) continue;
        if ((img[bp * N + bq] - bg[bp * N + bq]) / sigma < thr) continue;
        out.push(describe(bp, bq, [r0, c0]));
      }
      return out;
    }

    // climb uphill from a clicked pixel to the nearest local peak (at most `steps` moves)
    function probe(r, c, steps) {
      let p = r, q = c;
      for (let k = 0; k < steps; k++) {
        let bp = p, bq = q, best = img[p * N + q];
        for (let a = Math.max(p - 1, 0); a <= Math.min(p + 1, N - 1); a++)
          for (let b = Math.max(q - 1, 0); b <= Math.min(q + 1, N - 1); b++)
            if (img[a * N + b] > best) { best = img[a * N + b]; bp = a; bq = b; }
        if (bp === p && bq === q) break;
        p = bp; q = bq;
      }
      return describe(p, q, null);
    }

    return { N, img, es, W, sigma, bg, scan, probe, classify };
  }

  const api = { build };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Detector = api;
})(typeof window !== "undefined" ? window : globalThis);
