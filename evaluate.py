"""Unified evaluation of every model on held-out slow-ramp runs and fast lean-out runs."""
import json
import numpy as np
from common import (load_run, run_ids, meta, summarize, decide, tune_tau, RES, run_events)
from train_classical import rms_rule, rms_danger, smooth_probs

M = {r["run_id"]: r for r in meta()["runs"]}
cl = json.load(open(RES / "classical.json"))
lat_feat = cl["latency"]["features"]

def lat_of(k):
    try:
        return json.load(open(RES / f"{k}_train.json"))["latency_ms"]
    except FileNotFoundError:
        return None

PROB_MODELS = [
    ("svm", "SVM (5 features)", lambda: cl["latency"]["svm"] + lat_feat),
    ("rf", "Random Forest (5 features)", lambda: cl["latency"]["rf"] + lat_feat),
    ("rf_smooth", "Random Forest + 100 ms smoothing", lambda: cl["latency"]["rf"] + lat_feat),
    ("cnn", "1D-CNN (raw p′)", lambda: lat_of("cnn")),
    ("lstm_p", "CNN-LSTM (raw p′)", lambda: lat_of("lstm_p")),
    ("lstm_pq", "CNN-LSTM (p′ + q′ fusion)", lambda: lat_of("lstm_pq")),
]


def load_probs(key):
    src = "rf" if key == "rf_smooth" else key
    try:
        z = np.load(RES / "probs" / f"{src}.npz")
    except FileNotFoundError:
        return None
    pr = {int(k): z[k] for k in z.files}
    if key == "rf_smooth":
        pr = {k: smooth_probs(v) for k, v in pr.items()}
    return pr


def ys_ons(split):
    ys, ons, F = {}, {}, {}
    for r in run_ids(split):
        d = load_run(r); ys[r] = d["y"]; ons[r] = M[r]["onset_idx"]; F[r] = d["F"]
    return ys, ons, F


def quiet_gap(preds, ys, F):
    """Recall on true-Transition windows, split by whether the window holds a burst (rms>=0.3) or a calm gap."""
    burst_hit = burst_n = gap_hit = gap_n = flips = 0
    t_trans = 0
    for r in preds:
        y, p, rms = ys[r], preds[r], F[r][:, 0]
        m = y == 1
        b = m & (rms >= 0.3); g = m & (rms < 0.3)
        burst_hit += (p[b] >= 1).sum(); burst_n += b.sum()
        gap_hit += (p[g] >= 1).sum(); gap_n += g.sum()
        pm = p[m] >= 1
        flips += int((pm[1:] != pm[:-1]).sum()); t_trans += m.sum() * 0.01
    return dict(burst_recall=float(burst_hit / max(burst_n, 1)), gap_recall=float(gap_hit / max(gap_n, 1)),
                gap_fraction=float(gap_n / max(gap_n + burst_n, 1)), flips_per_s=float(flips / max(t_trans, 1e-9)))


if __name__ == "__main__":
    S = {s: ys_ons(s) for s in ["val", "test", "fast"]}
    out = {"default": {"test": [], "fast": []}, "tuned": {"test": [], "fast": []}, "tau": {}, "tradeoff": {},
           "quiet_gap": {}}
    # hard-rule baselines (no threshold to tune)
    for name, fn in [("RMS danger alarm (1.2 kPa)", rms_danger), ("RMS 3-level threshold", rms_rule)]:
        for split in ["test", "fast"]:
            ys, ons, F = S[split]
            pr = {r: fn(F[r]) for r in ys}
            s = summarize(name, pr, ys, ons, latency_ms=0.02, extra=dict(key=name, kind="rule"))
            out["default"][split].append(s); out["tuned"][split].append(s)
            if split == "test":
                out["quiet_gap"][name] = quiet_gap(pr, ys, F)
    for key, name, latf in PROB_MODELS:
        probs = load_probs(key)
        if probs is None:
            print("skip", key); continue
        yv, _, _ = S["val"]
        tau = tune_tau([probs[r] for r in yv], [yv[r] for r in yv], max_far=0.05)
        out["tau"][key] = tau
        lat = latf()
        for mode in ["default", "tuned"]:
            for split in ["test", "fast"]:
                ys, ons, F = S[split]
                pr = {r: (probs[r].argmax(1) if mode == "default" else decide(probs[r], tau)) for r in ys}
                s = summarize(name, pr, ys, ons, latency_ms=lat, extra=dict(key=key, kind="ml", tau=tau if mode == "tuned" else None))
                out[mode][split].append(s)
                if split == "test" and mode == "tuned":
                    out["quiet_gap"][name] = quiet_gap(pr, ys, F)
        # trade-off curve on test + fast
        curve = []
        for t in np.linspace(0.3, 0.995, 40):
            row = {"tau": float(t)}
            for split in ["test", "fast"]:
                ys, ons, F = S[split]
                pr = {r: decide(probs[r], t) for r in ys}
                s = summarize(name, pr, ys, ons)
                row[split] = dict(far=s["false_alarm_rate"], acc=s["accuracy"], recall_T=s["recall"][1],
                                  lead_med=s["lead_ms_median"], fa_min=s["false_alarms_per_min"])
            curve.append(row)
        out["tradeoff"][key] = curve
    json.dump(out, open(RES / "summary.json", "w"), indent=1)
    for mode in ["default", "tuned"]:
        for split in ["test", "fast"]:
            print(f"--- {mode} / {split}")
            for s in out[mode][split]:
                print(f"{s['model']:34s} acc {s['accuracy']:.3f} F1 {s['macro_f1']:.3f} recT {s['recall'][1]:.2f} FAR {s['false_alarm_rate']:.3f} "
                      f"lead med {s['lead_ms_median']:.0f} min {s['lead_ms_min']:.0f} >=40 {s['pct_runs_lead_ge_valve']:.2f} FA/min {s['false_alarms_per_min']:.2f} lat {s.get('latency_ms') or 0:.2f}")
    print(json.dumps(out["quiet_gap"], indent=0))
    print(out["tau"])
