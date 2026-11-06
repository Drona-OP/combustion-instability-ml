"""Pack everything the website needs into web/data.js (window.DATA = {...})."""
import base64, json
import numpy as np
from scipy.signal import decimate, stft
from common import load_run, run_ids, meta, decide, alarm_track, RES, ROOT, CLASSES
from train_classical import rms_rule, rms_danger, smooth_probs
from features import FEATURE_NAMES, FEATURE_LABELS

OUTDIR = ROOT / "web"; OUTDIR.mkdir(exist_ok=True)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
S = json.load(open(RES / "summary.json"))
CL = json.load(open(RES / "classical.json"))
M = meta(); MR = {r["run_id"]: r for r in M["runs"]}

MODEL_KEYS = ["rms_danger", "rms_rule", "svm", "rf", "rf_smooth", "cnn", "lstm_p", "lstm_pq"]
MODEL_NAMES = {"rms_danger": "RMS danger alarm (1.2 kPa)", "rms_rule": "RMS 3-level threshold",
               "svm": "SVM (5 features)", "rf": "Random Forest (5 features)",
               "rf_smooth": "Random Forest + 100 ms smoothing", "cnn": "1D-CNN (raw p′)",
               "lstm_p": "CNN-LSTM (raw p′)", "lstm_pq": "CNN-LSTM (p′ + q′ fusion)"}


def probs_of(key):
    src = "rf" if key == "rf_smooth" else key
    p = RES / "probs" / f"{src}.npz"
    if not p.exists():
        return None
    z = np.load(p); d = {int(k): z[k] for k in z.files}
    return {k: smooth_probs(v) for k, v in d.items()} if key == "rf_smooth" else d


PROBS = {k: probs_of(k) for k in MODEL_KEYS[2:]}
AVAILABLE = [k for k in MODEL_KEYS if k in ("rms_danger", "rms_rule") or PROBS.get(k) is not None]


def replay_run(r):
    d = load_run(r)
    F = d["F"]; y = d["y"]; n = len(y)
    p25 = decimate(d["p"].astype(np.float64), 4, ftype="fir"); q25 = decimate(d["q"].astype(np.float64), 4, ftype="fir")
    rec = dict(id=r, kind=MR[r]["kind"], duration=MR[r]["duration"], onset_idx=MR[r]["onset_idx"],
               phi_ST=MR[r]["phi_ST"], phi_TU=MR[r]["phi_TU"], n=n, fs_plot=2500,
               p=b64(np.clip(np.round(p25 * 1000), -32767, 32767).astype(np.int16)),
               q=b64(np.clip(np.round(q25 * 1000), -32767, 32767).astype(np.int16)),
               phi=b64(np.round(d["phi"][d["last"]] * 10000).astype(np.int16)),
               y=b64(y.astype(np.uint8)), rms=b64(np.round(F[:, 0] * 1000).astype(np.int16)),
               feat=b64(F.astype(np.float32)),
               pred={}, pns={}, alarm={}, lead={})
    for k in AVAILABLE:
        if k == "rms_rule":
            pr = rms_rule(F)
        elif k == "rms_danger":
            pr = rms_danger(F)
        else:
            P = PROBS[k][r]; pr = decide(P, S["tau"][k])
            rec["pns"][k] = b64(np.round((P[:, 1] + P[:, 2]) * 255).astype(np.uint8))
        pr = pr.astype(np.uint8)
        rec["pred"][k] = b64(pr); rec["alarm"][k] = b64(alarm_track(pr).astype(np.uint8))
        row = next((s for s in S["tuned"]["test" if MR[r]["kind"] == "ramp" else "fast"] if s["model"] == MODEL_NAMES[k]), None)
        if row:
            rec["lead"][k] = row["per_run"][str(r)]["lead_ms"]
    # spectrogram of p' (0-800 Hz, 100 ms frames, 50 ms hop), dB scaled to uint8
    f, t, Z = stft(d["p"].astype(np.float64), fs=10000, nperseg=1024, noverlap=512, boundary=None)
    sel = f <= 800
    db = 10 * np.log10(np.abs(Z[sel]) ** 2 + 1e-9)
    lo, hi = -75.0, -5.0
    rec["spec"] = dict(nf=int(sel.sum()), nt=int(Z.shape[1]), fmax=float(f[sel][-1]), t0=float(t[0]), dt=float(t[1] - t[0]),
                       lo=lo, hi=hi, z=b64(np.clip((db - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8).T))
    return rec


def bifurcation():
    pts = []
    for r in run_ids("train") + run_ids("val") + run_ids("test"):
        d = load_run(r)
        idx = np.arange(0, len(d["y"]), 4)
        pts.append(np.stack([d["phi"][d["last"][idx]], d["F"][idx, 0], d["y"][idx]], 1))
    P = np.concatenate(pts)
    return dict(phi=b64(np.round(P[:, 0] * 10000).astype(np.int16)), rms=b64(np.round(P[:, 1] * 1000).astype(np.int16)),
                y=b64(P[:, 2].astype(np.uint8)), n=len(P))


def feature_hists():
    X, Y = [], []
    for r in run_ids("train"):
        d = load_run(r); X.append(d["F"]); Y.append(d["y"])
    X = np.concatenate(X); Y = np.concatenate(Y)
    out = []
    ranges = {"p_rms": (0, 2.6), "kurtosis": (1, 8), "peak_ratio": (0, 1), "zcr_reg": (0.6, 2.4), "phase_lag": (0, 180)}
    for j, name in enumerate(FEATURE_NAMES):
        lo, hi = ranges[name]; bins = np.linspace(lo, hi, 41)
        hs = [np.histogram(np.clip(X[Y == c, j], lo, hi - 1e-9), bins=bins)[0] / max((Y == c).sum(), 1) for c in range(3)]
        med = [float(np.median(X[Y == c, j])) for c in range(3)]
        out.append(dict(name=name, label=FEATURE_LABELS[j], lo=lo, hi=hi, h=[h.round(5).tolist() for h in hs], median=med))
    return out


def rayleigh():
    """Normalised Rayleigh index <p'q'>/(p_rms q_rms) per regime, waveguide delay removed."""
    acc = {0: [], 1: [], 2: []}
    for r in run_ids("test") + run_ids("val"):
        d = load_run(r)
        p = d["p"][5:].astype(np.float64); q = d["q"][:-5].astype(np.float64); reg = d["regime"][:-5]
        for s in range(0, len(p) - 500, 500):
            seg = slice(s, s + 500)
            pp = p[seg] - p[seg].mean(); qq = q[seg] - q[seg].mean()
            acc[int(reg[s + 250])].append((pp * qq).mean() / (pp.std() * qq.std() + 1e-12))
    return [dict(regime=CLASSES[c], mean=float(np.mean(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), n=len(v)) for c, v in acc.items()]


def phase_vs_phi():
    d = load_run(14); idx = np.arange(0, len(d["y"]), 10)
    return dict(phi=np.round(d["phi"][d["last"][idx]], 4).tolist(), phase=np.round(d["F"][idx, 4], 1).tolist(),
                y=d["y"][idx].tolist())


def far_by_distance():
    """False-alarm rate on truly-stable windows, binned by how far phi is above the run's Stable/Transition boundary."""
    bins = [0, 0.01, 0.02, 0.04, 0.5]
    out = {"bins": bins, "models": {}}
    for k in AVAILABLE:
        D, Fl = [], []
        for r in run_ids("test") + run_ids("fast"):
            d = load_run(r); y = d["y"]; st = y == 0
            if k == "rms_rule": pr = rms_rule(d["F"])
            elif k == "rms_danger": pr = rms_danger(d["F"])
            else: pr = decide(PROBS[k][r], S["tau"][k])
            D.append(d["phi"][d["last"]][st] - MR[r]["phi_ST"]); Fl.append(pr[st] >= 1)
        D = np.concatenate(D); Fl = np.concatenate(Fl)
        out["models"][k] = [float(Fl[(D >= a) & (D < b)].mean()) for a, b in zip(bins[:-1], bins[1:])]
        out["n"] = [int(((D >= a) & (D < b)).sum()) for a, b in zip(bins[:-1], bins[1:])]
    return out


def trim(rows):
    keep = ["model", "key", "kind", "accuracy", "macro_f1", "recall", "precision", "f1", "false_alarm_rate", "confusion",
            "lead_ms_median", "lead_ms_min", "lead_ms_mean", "pct_runs_lead_ge_valve", "false_alarm_episodes",
            "false_alarms_per_min", "latency_ms", "per_run", "tau"]
    return [{k: r.get(k) for k in keep} for r in rows]


if __name__ == "__main__":
    deep_info = {}
    for k in ["cnn", "lstm_p", "lstm_pq"]:
        p = RES / f"{k}_train.json"
        if p.exists():
            j = json.load(open(p)); deep_info[k] = dict(n_params=j["n_params"], latency_ms=j["latency_ms"], train_time_s=j["train_time_s"],
                                                       best_val_acc=j["best_val_acc"], history=j["history"])
    data = dict(
        models=[dict(key=k, name=MODEL_NAMES[k]) for k in AVAILABLE],
        runs=M["runs"], split=M["split"],
        summary=dict(default={s: trim(v) for s, v in S["default"].items()}, tuned={s: trim(v) for s, v in S["tuned"].items()},
                     tau=S["tau"], tradeoff=S["tradeoff"], quiet_gap=S["quiet_gap"]),
        importance=CL["rf_importance"], window_study=CL["window_study"], latency=CL["latency"], deep=deep_info,
        real=json.load(open(RES / "real_check.json")), real_examples=json.load(open(RES / "real_examples.json")),
        bif=bifurcation(), far_dist=far_by_distance(), hists=feature_hists(), rayleigh=rayleigh(), phase=phase_vs_phi(),
        forest=json.load(open(RES / "browser_forest.json")),
        replay=[replay_run(r) for r in run_ids("test") + run_ids("fast")],
    )
    s = "window.DATA=" + json.dumps(data, separators=(",", ":")) + ";"
    (OUTDIR / "data.js").write_text(s)
    print("models", AVAILABLE, "data.js MB", round(len(s) / 1e6, 2))
    print("rayleigh", data["rayleigh"])
