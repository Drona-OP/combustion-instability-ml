"""RMS-threshold baselines, SVM, Random Forest (+ smoothed RF), window-size study."""
import json, time, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import load_run, run_ids, meta, summarize, RES, ROOT
from features import FEATURE_NAMES, make_windows, window_features

M = {r["run_id"]: r for r in meta()["runs"]}


def gather(split):
    X, Y, runs = [], [], []
    for r in run_ids(split):
        d = load_run(r)
        X.append(d["F"]); Y.append(d["y"]); runs.append(np.full(len(d["y"]), r))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(runs)


def rms_rule(F):
    r = F[:, 0]
    return np.where(r < 0.3, 0, np.where(r < 1.2, 1, 2))


def rms_danger(F):
    # industry practice: a single alarm level at the danger amplitude (1.2 kPa)
    return np.where(F[:, 0] >= 1.2, 2, 0)


def smooth_probs(proba, k=10):
    """Causal moving average of class probabilities over the last k windows (report 7.2 fix)."""
    c = np.cumsum(np.vstack([np.zeros((1, proba.shape[1])), proba]), axis=0)
    out = np.empty_like(proba)
    for i in range(len(proba)):
        j = max(0, i + 1 - k)
        out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out


def per_run_preds(fn, split):
    preds, ys, ons = {}, {}, {}
    for r in run_ids(split):
        d = load_run(r)
        preds[r] = fn(d["F"]).astype(int); ys[r] = d["y"]; ons[r] = M[r]["onset_idx"]
    return preds, ys, ons


def latency(fn, F, n=300):
    t = time.perf_counter()
    for i in range(n):
        fn(F[i:i + 1])
    return (time.perf_counter() - t) / n * 1000


def feat_latency(n=300):
    d = load_run(14)
    t = time.perf_counter()
    for i in range(n):
        window_features(d["p"][None, i * 100:i * 100 + 500], d["q"][None, i * 100:i * 100 + 500])
    return (time.perf_counter() - t) / n * 1000


if __name__ == "__main__":
    Xtr, ytr, _ = gather("train")
    Xva, yva, _ = gather("val")
    print("train", Xtr.shape, np.bincount(ytr), "val", Xva.shape)

    svm = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0))
    t = time.time(); svm.fit(Xtr[::2], ytr[::2]); print("svm fit", time.time() - t)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    print("val acc svm %.3f rf %.3f" % ((svm.predict(Xva) == yva).mean(), (rf.predict(Xva) == yva).mean()))

    fl = feat_latency()
    models = {
        "RMS danger alarm (1.2 kPa)": (rms_danger, lambda F: rms_danger(F)),
        "RMS 3-level threshold": (rms_rule, lambda F: rms_rule(F)),
        "SVM (5 features)": (lambda F: svm.predict(F), lambda F: svm.predict(F)),
        "Random Forest (5 features)": (lambda F: rf.predict(F), lambda F: rf.predict(F)),
        "Random Forest + 100 ms smoothing": (lambda F: smooth_probs(rf.predict_proba(F)).argmax(1), lambda F: rf.predict_proba(F)),
    }
    Fte = gather("test")[0]
    results = {}
    for split in ["test", "fast"]:
        results[split] = []
        for name, (fn, single) in models.items():
            pr, ys, ons = per_run_preds(fn, split)
            lat = fl + (latency(single, Fte) if "RMS" not in name else 0.0)
            if "RMS" in name:
                lat = latency(lambda F: None, Fte) + 0.02  # rms only ~ trivial
            s = summarize(name, pr, ys, ons, latency_ms=lat)
            results[split].append(s)
            print(f"[{split}] {name:34s} acc {s['accuracy']:.3f} F1 {s['macro_f1']:.3f} FAR {s['false_alarm_rate']:.3f} "
                  f"lead med {s['lead_ms_median']:.0f} min {s['lead_ms_min']:.0f} >=40ms {s['pct_runs_lead_ge_valve']:.2f} "
                  f"FA/min {s['false_alarms_per_min']:.2f} lat {lat:.2f}ms")

    imp = dict(zip(FEATURE_NAMES, rf.feature_importances_.tolist()))
    print("RF importances", imp)

    # ---- window-size study (RF) ------------------------------------------------
    ws_res = []
    for win in [300, 500, 1000]:
        def feats(split):
            X, Y, R = [], [], []
            for r in run_ids(split):
                d = load_run(r)
                Wp, last = make_windows(d["p"], win, 100); Wq, _ = make_windows(d["q"], win, 100)
                F, _ = window_features(Wp, Wq)
                X.append(F); Y.append(d["regime"][last]); R.append(r)
            return X, Y, R
        Xa, Ya, _ = feats("train")
        m = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(np.concatenate(Xa), np.concatenate(Ya))
        out = {}
        for split in ["test", "fast"]:
            Xb, Yb, Rb = feats(split)
            pr = {r: m.predict(x) for r, x in zip(Rb, Xb)}; ys = dict(zip(Rb, Yb))
            # onset index recomputed on this window grid
            ons = {}
            for r, x in zip(Rb, Xb):
                hi = x[:, 0] >= 1.2
                sus = np.array([hi[i] and hi[i:i + 30].mean() >= 0.9 for i in range(len(hi))])
                ons[r] = int(np.argmax(sus)) if sus.any() else -1
            out[split] = summarize(f"RF {win/10:.0f} ms", pr, ys, ons)
        ws_res.append(dict(window_ms=win / 10, test_acc=out["test"]["accuracy"], test_f1=out["test"]["macro_f1"],
                           test_far=out["test"]["false_alarm_rate"], fast_acc=out["fast"]["accuracy"],
                           fast_lead_med=out["fast"]["lead_ms_median"], test_lead_med=out["test"]["lead_ms_median"]))
        print("window", win / 10, ws_res[-1])

    # save per-run class probabilities for unified evaluation
    import os
    (RES / "probs").mkdir(exist_ok=True)
    rf.set_params(n_jobs=1)
    for mname, mdl in [("svm", svm), ("rf", rf)]:
        out = {}
        for split in ["val", "test", "fast"]:
            for r in run_ids(split):
                out[str(r)] = mdl.predict_proba(load_run(r)["F"]).astype(np.float32)
        np.savez_compressed(RES / "probs" / f"{mname}.npz", **out)
    lat_rf = latency(lambda F: rf.predict_proba(F), Fte); lat_svm = latency(lambda F: svm.predict_proba(F), Fte)
    print("single-window latency (ms) feat %.2f rf %.2f svm %.2f" % (fl, lat_rf, lat_svm))
    json.dump(dict(results=results, rf_importance=imp, window_study=ws_res,
                   latency=dict(features=fl, rf=lat_rf, svm=lat_svm)), open(RES / "classical.json", "w"), indent=1)
    pickle.dump(dict(rf=rf, svm=svm), open(RES / "classical_models.pkl", "wb"))
