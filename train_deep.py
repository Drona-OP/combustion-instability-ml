"""
Deep models in JAX / Flax (CPU):
  * 1D-CNN      : raw 50 ms p' window -> class                      (no memory)
  * CNN-LSTM    : last 16 windows (200 ms of history), each encoded by a 1D-CNN,
                  fed through an LSTM -> class of the newest window
                  variants: p' only, and p'+q' sensor fusion (2 channels)
Usage: python train_deep.py cnn | lstm_p | lstm_pq
"""
import sys, time, json, pickle
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from common import load_run, run_ids, meta, RES

WIN, STRIDE, L = 500, 100, 16
P_SCALE, Q_SCALE = 1.0, 1.0


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):  # x: (N, 500, C)
        for feat, k, s in [(16, 9, 2), (32, 7, 2), (64, 5, 2), (64, 5, 2)]:
            x = nn.Conv(feat, (k,), strides=(s,), padding="SAME")(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        mean = x.mean(axis=1)
        mx = x.max(axis=1)
        return nn.Dense(64)(jnp.concatenate([mean, mx], -1))


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):  # (B, 500, C)
        h = nn.gelu(Encoder()(x))
        return nn.Dense(3)(nn.Dense(64)(h))


class CNNLSTM(nn.Module):
    @nn.compact
    def __call__(self, x):  # (B, T, 500, C) windows; returns logits for each t >= L-1 using last L windows
        B, T = x.shape[:2]
        e = Encoder()(x.reshape((B * T,) + x.shape[2:])).reshape(B, T, -1)
        e = nn.gelu(e)
        n_out = T - L + 1
        idx = jnp.arange(n_out)[:, None] + jnp.arange(L)[None, :]  # (n_out, L)
        seq = e[:, idx]  # (B, n_out, L, D)
        seq = seq.reshape(B * n_out, L, -1)
        h = nn.RNN(nn.OptimizedLSTMCell(64))(seq)[:, -1]
        return nn.Dense(3)(nn.gelu(nn.Dense(64)(h))).reshape(B, n_out, 3)


def windows_of(sig, start_w, n_w):
    i0 = start_w * STRIDE
    idx = np.arange(WIN)[None, :] + STRIDE * np.arange(n_w)[:, None]
    return sig[i0:][idx]


def load_split(split):
    return {r: load_run(r) for r in run_ids(split)}


def make_input(d, chans, start_w, n_w):
    ch = []
    if "p" in chans: ch.append(windows_of(d["p"], start_w, n_w) / P_SCALE)
    if "q" in chans: ch.append(windows_of(d["q"], start_w, n_w) / Q_SCALE)
    return np.stack(ch, -1).astype(np.float32)


def train(kind, steps=2500, seed=0):
    chans = "pq" if kind == "lstm_pq" else "p"
    seq = kind.startswith("lstm")
    model = CNNLSTM() if seq else CNN()
    tr, va = load_split("train"), load_split("val")
    rng = np.random.default_rng(seed)
    C = 32  # decisions per chunk
    B = 8 if seq else 256
    def batch():
        if seq:
            xs, ys = [], []
            for _ in range(B):
                r = rng.choice(list(tr)); d = tr[r]
                nW = len(d["y"]); s = rng.integers(0, nW - (C + L - 1))
                xs.append(make_input(d, chans, s, C + L - 1)); ys.append(d["y"][s + L - 1: s + L - 1 + C])
            return np.stack(xs), np.stack(ys)
        xs, ys = [], []
        for _ in range(8):
            r = rng.choice(list(tr)); d = tr[r]
            w = rng.integers(0, len(d["y"]), B // 8)
            xs.append(np.concatenate([make_input(d, chans, int(i), 1) for i in w])); ys.append(d["y"][w])
        return np.concatenate(xs), np.concatenate(ys)

    x0, _ = batch()
    params = model.init(jax.random.PRNGKey(seed), x0[:1])
    sched = optax.warmup_cosine_decay_schedule(0, 2e-3, min(100, steps // 4), steps, 1e-5)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    opt = tx.init(params)

    @jax.jit
    def step(params, opt, x, y):
        def lossf(p):
            lg = model.apply(p, x)
            return optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
        l, g = jax.value_and_grad(lossf)(params)
        up, opt = tx.update(g, opt, params)
        return optax.apply_updates(params, up), opt, l

    fwd = jax.jit(lambda p, x: jax.nn.softmax(model.apply(p, x), -1))

    def predict_run(params, d):
        nW = len(d["y"])
        if seq:
            # pad with first window so early decisions have L windows of context
            out = []
            chunk = 256
            for s in range(0, nW, chunk):
                n = min(chunk, nW - s)
                st = s - (L - 1)
                x = make_input(d, chans, max(st, 0), n + L - 1 - max(0, -st))
                if st < 0:
                    x = np.concatenate([np.repeat(x[:1], -st, 0), x], 0)
                out.append(np.asarray(fwd(params, x[None]))[0])
            return np.concatenate(out)
        out = []
        for s in range(0, nW, 1024):
            out.append(np.asarray(fwd(params, make_input(d, chans, s, min(1024, nW - s)))))
        return np.concatenate(out)

    t0 = time.time(); best = (-1, None); hist = []
    for it in range(1, steps + 1):
        x, y = batch()
        params, opt, l = step(params, opt, x, y)
        if it % 250 == 0 or it == steps:  # validation check
            accs = [(predict_run(params, d).argmax(1) == d["y"]).mean() for d in va.values()]
            va_acc = float(np.mean(accs)); hist.append(dict(step=it, loss=float(l), val_acc=va_acc))
            print(f"{kind} step {it} loss {float(l):.3f} val acc {va_acc:.3f}  {time.time()-t0:.0f}s", flush=True)
            if va_acc > best[0]:
                best = (va_acc, jax.tree_util.tree_map(np.asarray, params))
    params = best[1]
    out = {}
    for split in ["val", "test", "fast"]:
        for r in run_ids(split):
            out[str(r)] = predict_run(params, load_run(r)).astype(np.float32)
    (RES / "probs").mkdir(exist_ok=True)
    np.savez_compressed(RES / "probs" / f"{kind}.npz", **out)
    pickle.dump(params, open(RES / f"{kind}_params.pkl", "wb"))

    # single-decision latency (one new window; LSTM re-encodes the full 200 ms context = worst case)
    d = load_run(14)
    x1 = make_input(d, chans, 1000, L if seq else 1)
    x1 = x1[None] if seq else x1
    fwd(params, x1).block_until_ready()
    t = time.perf_counter()
    for _ in range(200):
        fwd(params, x1).block_until_ready()
    lat = (time.perf_counter() - t) / 200 * 1000
    n_par = int(sum(np.size(v) for v in jax.tree_util.tree_leaves(params)))
    json.dump(dict(kind=kind, history=hist, best_val_acc=best[0], latency_ms=lat, n_params=n_par,
                   train_time_s=time.time() - t0), open(RES / f"{kind}_train.json", "w"), indent=1)
    print(kind, "done; latency %.2f ms, params %d" % (lat, n_par))


if __name__ == "__main__":
    train(sys.argv[1], steps=int(sys.argv[2]) if len(sys.argv) > 2 else 2500)
