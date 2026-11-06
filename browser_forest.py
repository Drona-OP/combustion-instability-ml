"""Compact Random Forest for in-browser inference (60 trees, depth <= 10) + its tuned threshold."""
import base64, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from common import load_run, run_ids, tune_tau, decide, RES

def gather(split):
    X, Y = [], []
    for r in run_ids(split):
        d = load_run(r); X.append(d["F"]); Y.append(d["y"])
    return X, Y

if __name__ == "__main__":
    Xtr, Ytr = gather("train")
    rf = RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=5, random_state=0, n_jobs=-1)
    rf.fit(np.concatenate(Xtr), np.concatenate(Ytr))
    Xv, Yv = gather("val")
    tau = tune_tau([rf.predict_proba(x) for x in Xv], Yv)
    Xt, Yt = gather("test")
    acc = float(np.mean(np.concatenate([decide(rf.predict_proba(x), tau) == y for x, y in zip(Xt, Yt)])))
    feat, thr, left, right, val, roots = [], [], [], [], [], []
    off = 0
    for est in rf.estimators_:
        t = est.tree_
        roots.append(off)
        f = t.feature.copy(); f[t.children_left == -1] = -1
        feat.append(f.astype(np.int8)); thr.append(t.threshold.astype(np.float32))
        l = np.where(t.children_left >= 0, t.children_left + off, -1); r = np.where(t.children_right >= 0, t.children_right + off, -1)
        left.append(l.astype(np.int32)); right.append(r.astype(np.int32))
        v = t.value[:, 0, :]; v = v / v.sum(1, keepdims=True)
        val.append(np.round(v * 255).astype(np.uint8).ravel())
        off += t.node_count
    enc = lambda a: base64.b64encode(np.concatenate(a).tobytes()).decode()
    out = dict(feat=enc(feat), thr=enc(thr), left=enc(left), right=enc(right), val=enc(val), roots=roots,
               tau=tau, n_nodes=off, test_acc_tuned=acc)
    json.dump(out, open(RES / "browser_forest.json", "w"))
    print("nodes", off, "tau", tau, "test acc (tuned)", acc, "size KB", len(json.dumps(out)) // 1024)
    # reference probabilities for a parity test
    d = load_run(14)
    np.save(RES / "parity_proba.npy", rf.predict_proba(d["F"][:400]))
