"""All figures for the LaTeX report, rendered from the project's actual data and results (vector PDF)."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root (report/scripts/ -> repo)
REPORT = Path(__file__).resolve().parents[1]

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(REPO))
import simulator as S
from common import load_run, run_ids, meta, decide, alarm_track
from scipy.signal import welch
import openpyxl

OUT = str(REPORT / "figures") + "/"
R = str(REPO / "results") + "/"
SUM = json.load(open(R + "summary.json")); CL = json.load(open(R + "classical.json")); RC = json.load(open(R + "real_check.json"))
M = {r["run_id"]: r for r in meta()["runs"]}
C0, C1, C2 = "#1f77b4", "#ff7f0e", "#d62728"
CC = [C0, C1, C2]; NAMES = ["Stable", "Transition", "Unstable"]
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "legend.frameon": False, "figure.dpi": 150})
def save(fig, name): fig.savefig(OUT + name + ".pdf"); plt.close(fig)
def row(name, scn="fast"): return [r for r in SUM["tuned"][scn] if r["model"] == name][0]
MN = {"rms_danger": "RMS danger alarm (1.2 kPa)", "rms_rule": "RMS 3-level threshold", "svm": "SVM (5 features)", "rf": "Random Forest (5 features)",
      "rf_smooth": "Random Forest + 100 ms smoothing", "cnn": "1D-CNN (raw p′)", "lstm_p": "CNN-LSTM (raw p′)", "lstm_pq": "CNN-LSTM (p′ + q′ fusion)"}
SHORT = {"rms_danger": "RMS danger", "rms_rule": "RMS 3-level", "svm": "SVM", "rf": "Random Forest", "rf_smooth": "RF + smoothing", "cnn": "1D-CNN", "lstm_p": "CNN-LSTM (p′)", "lstm_pq": "CNN-LSTM (p′+q′)"}

d14 = load_run(14)

# 1. example signals p' and q' in the three regimes
fig, axs = plt.subplots(3, 1, figsize=(7, 5.6), sharex=True)
for ax, (ph, c) in zip(axs, [(0.86, 0), (0.695, 1), (0.60, 2)]):
    k = int(np.argmin(abs(d14["phi"] - ph))); t = np.arange(2000) / 10
    ax.plot(t, d14["p"][k:k + 2000], color=CC[c], lw=0.8, label="p′ (kPa)")
    ax.plot(t, d14["q"][k:k + 2000] - 0.0, color="0.45", lw=0.6, alpha=0.8, label="q′ (a.u.)")
    ax.set_ylim(-4, 4); ax.set_ylabel("signal")
    ax.set_title(f"{NAMES[c]}, φ = {ph:.2f}", fontsize=10, loc="left")
axs[0].legend(loc="upper right", ncol=2, fontsize=9); axs[-1].set_xlabel("time (ms)")
save(fig, "signals")

# 2. spectra
fig, ax = plt.subplots(figsize=(4.4, 3.2))
for ph, c in [(0.86, 0), (0.695, 1), (0.60, 2)]:
    k = int(np.argmin(abs(d14["phi"] - ph))); f, P = welch(d14["p"][k:k + 10000], fs=10000, nperseg=2048)
    ax.semilogy(f, P, color=CC[c], lw=1.1, label=f"{NAMES[c]} (φ = {ph:.2f})")
ax.set_xlim(0, 1000); ax.legend(fontsize=8); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD of p′ (kPa²/Hz)"); ax.legend(fontsize=9)
ax.axvline(210, color="0.5", ls="--", lw=0.8); ax.text(215, ax.get_ylim()[1] * 0.3, "quarter-wave mode", fontsize=8, color="0.3")
save(fig, "spectra")

# 3. theta(phi) and nu(phi) for the reference rig
P0 = S.RunParams(run_id=0, seed=0); phi = np.linspace(0.55, 0.90, 300)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
axs[0].plot(phi, P0.theta_deg(phi), "k", lw=1.4); axs[0].axhline(90, color="0.5", ls="--", lw=0.8); axs[0].text(0.56, 93, "90°: Rayleigh boundary", fontsize=8)
axs[0].set_xlabel("equivalence ratio φ"); axs[0].set_ylabel("flame phase lag θ (deg)"); axs[0].invert_xaxis()
nu = P0.nu(phi); axs[1].plot(phi, nu, "k", lw=1.4)
b1, b2 = P0.boundaries_phi()
axs[1].axhspan(-100, P0.nu_T, color=C0, alpha=0.12); axs[1].axhspan(P0.nu_T, 0, color=C1, alpha=0.15); axs[1].axhspan(0, 30, color=C2, alpha=0.12)
axs[1].text(0.89, -85, "Stable", fontsize=8); axs[1].text(0.89, -30, "Transition", fontsize=8); axs[1].text(0.89, 8, "Unstable", fontsize=8)
axs[1].set_ylim(-95, 28); axs[1].set_xlabel("equivalence ratio φ"); axs[1].set_ylabel("mean growth rate ν (s⁻¹)"); axs[1].invert_xaxis()
fig.tight_layout(); save(fig, "theta_nu")

# 4. route to instability (all 15 sweeps)
fig, ax = plt.subplots(figsize=(6.8, 3.8))
for c in [0, 2, 1]:
    X, Y = [], []
    for r in run_ids("train") + run_ids("val") + run_ids("test"):
        d = load_run(r); idx = np.arange(0, len(d["y"]), 5); m = d["y"][idx] == c
        X += list(d["phi"][d["last"][idx]][m]); Y += list(d["F"][idx, 0][m])
    ax.scatter(X, Y, s=2, color=CC[c], alpha=0.5, label=NAMES[c], rasterized=True)
for v in (0.3, 1.2): ax.axhline(v, color="0.3", ls="--", lw=0.8)
ax.text(0.905, 0.33, "0.3 kPa", fontsize=8); ax.text(0.905, 1.23, "1.2 kPa", fontsize=8)
ax.set_xlim(0.91, 0.545); ax.set_ylim(0, 2.4); ax.set_xlabel("equivalence ratio φ (leaner →)"); ax.set_ylabel("window RMS of p′ (kPa)")
ax.legend(markerscale=5, fontsize=9, loc="upper left"); save(fig, "route")

# 5. Rayleigh index per regime (box plot)
vals = {0: [], 1: [], 2: []}
for r in run_ids("test") + run_ids("val"):
    d = load_run(r); p = d["p"][5:].astype(float); q = d["q"][:-5].astype(float); reg = d["regime"][:-5]
    for s0 in range(0, len(p) - 500, 500):
        pp = p[s0:s0 + 500] - p[s0:s0 + 500].mean(); qq = q[s0:s0 + 500] - q[s0:s0 + 500].mean()
        vals[int(reg[s0 + 250])].append((pp * qq).mean() / (pp.std() * qq.std() + 1e-12))
fig, ax = plt.subplots(figsize=(3.8, 3.2))
bp = ax.boxplot([vals[c] for c in range(3)], tick_labels=NAMES, patch_artist=True, widths=0.5, showfliers=False)
for patch, c in zip(bp["boxes"], CC): patch.set_facecolor(c); patch.set_alpha(0.35)
ax.axhline(0, color="0.3", lw=0.8); ax.set_ylabel("normalised Rayleigh index")
for c in range(3): ax.text(c + 1.28, np.median(vals[c]), f"{np.mean(vals[c]):+.2f}", fontsize=8, va="center")
save(fig, "rayleigh")
json.dump({str(c): [float(np.mean(v)), len(v)] for c, v in vals.items()}, open(OUT + "rayleigh.json", "w"))

# 6. feature distributions
X, Y = [], []
for r in run_ids("train"):
    d = load_run(r); X.append(d["F"]); Y.append(d["y"])
X = np.concatenate(X); Y = np.concatenate(Y)
labs = ["RMS of p′ (kPa)", "kurtosis", "FFT peak ratio", "zero-crossing regularity", "phase lag Δθ (deg)"]
rng = [(0, 2.6), (1, 8), (0, 1), (0.6, 2.4), (0, 180)]
fig, axs = plt.subplots(1, 5, figsize=(10, 2.4))
for j, ax in enumerate(axs):
    for c in range(3):
        ax.hist(np.clip(X[Y == c, j], *rng[j]), bins=40, range=rng[j], density=True, histtype="step", color=CC[c], lw=1.2, label=NAMES[c])
    ax.set_xlabel(labs[j], fontsize=8.5); ax.set_yticks([]); ax.tick_params(labelsize=8)
axs[0].legend(fontsize=7.5, loc="upper right"); fig.tight_layout(); save(fig, "features")

# 7. warning time per run (fast lean-outs)
order = ["rms_danger", "rms_rule", "svm", "rf", "rf_smooth", "cnn", "lstm_p", "lstm_pq"]
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, k in enumerate(order):
    r = row(MN[k]); v = np.array([x["lead_ms"] for x in r["per_run"].values()])
    ax.scatter(v, np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), s=14, color=C2 if k == "rms_danger" else ("#2ca02c" if k == "lstm_pq" else "0.35"), zorder=3)
    ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="k", lw=2)
ax.axvline(40, color=C2, ls="--", lw=1); ax.text(55, 7.45, "40 ms valve time", color=C2, fontsize=8)
ax.axvline(0, color="0.3", lw=0.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[k] for k in order]); ax.set_xlabel("warning time before full instability (ms)")
ax.grid(axis="y", alpha=0); save(fig, "leadtime")

# 8. quiet gap
o2 = ["rms_rule", "rf", "rf_smooth", "cnn", "lstm_p", "lstm_pq"]
Q = SUM["quiet_gap"]
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(len(o2)); w = 0.38
b = [Q[MN[k]]["burst_recall"] * 100 for k in o2]; g = [Q[MN[k]]["gap_recall"] * 100 for k in o2]
ax.bar(x - w / 2, b, w, color=C0, label="windows with a burst"); ax.bar(x + w / 2, g, w, color=C1, label="windows in a calm gap")
for i in range(len(o2)): ax.text(x[i] + w / 2, g[i] + 2, f"{g[i]:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in o2], fontsize=8.5, rotation=15, ha="right"); ax.set_ylabel("transition windows flagged (%)"); ax.set_ylim(0, 115); ax.legend(ncol=2, fontsize=9, loc="upper center")
save(fig, "quietgap")

# 9. confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
for ax, k in zip(axs, ["rf", "lstm_pq"]):
    cm = np.array(row(MN[k])["confusion"]); Pm = cm / cm.sum(1, keepdims=True)
    ax.imshow(Pm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{Pm[i, j] * 100:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8, color="w" if Pm[i, j] > 0.55 else "k")
    ax.set_xticks(range(3)); ax.set_xticklabels(NAMES, fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels(NAMES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(SHORT[k], fontsize=10); ax.grid(False)
fig.tight_layout(); save(fig, "confusion")

# 10. false alarms vs distance from boundary
def fard(k):
    z = np.load(R + f"probs/{k}.npz"); bins = [0, 0.01, 0.02, 0.04, 0.5]; D_, F_ = [], []
    for r in run_ids("test") + run_ids("fast"):
        d = load_run(r); st = d["y"] == 0; pr = decide(z[str(r)], SUM["tau"][k])
        D_.append(d["phi"][d["last"]][st] - M[r]["phi_ST"]); F_.append(pr[st] >= 1)
    D_ = np.concatenate(D_); F_ = np.concatenate(F_); return [F_[(D_ >= a) & (D_ < b)].mean() * 100 for a, b in zip(bins[:-1], bins[1:])]
fig, ax = plt.subplots(figsize=(4.3, 3.3)); xl = ["< 0.01", "0.01–0.02", "0.02–0.04", "> 0.04"]; x = np.arange(4)
fb, fr = fard("lstm_pq"), fard("rf")
ax.bar(x - 0.2, fr, 0.4, color="0.6", label="Random Forest"); ax.bar(x + 0.2, fb, 0.4, color="#2ca02c", label="CNN-LSTM (p′+q′)")
for i in range(4): ax.text(x[i] + 0.2, fb[i] + 0.8, f"{fb[i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=8); ax.set_xlabel("Δφ above the Stable→Transition boundary"); ax.set_ylabel("stable windows flagged (%)"); ax.legend(fontsize=9)
save(fig, "fardist"); json.dump({"lstm_pq": fb, "rf": fr}, open(OUT + "fard.json", "w"))

# 11. trade-off
fig, ax = plt.subplots(figsize=(4.4, 3.4))
for k, c in zip(["svm", "rf", "rf_smooth", "cnn", "lstm_p", "lstm_pq"], ["#1f77b4", "#ff7f0e", "0.55", "#9467bd", "#d62728", "#2ca02c"]):
    cur = SUM["tradeoff"][k]; xs = np.array([p["fast"]["far"] for p in cur]) * 100; ys = np.array([p["fast"]["recall_T"] for p in cur]) * 100; o = np.argsort(xs)
    ax.plot(xs[o], ys[o], color=c, lw=2 if k == "lstm_pq" else 1.3, label=SHORT[k])
rr = row(MN["rms_rule"]); ax.scatter([rr["false_alarm_rate"] * 100], [rr["recall"][1] * 100], marker="s", color="k", zorder=5, label="RMS 3-level")
ax.axvline(5, color="0.4", ls="--", lw=0.9); ax.set_xlim(0, 40); ax.set_ylim(0, 100)
ax.set_xlabel("false-alarm rate (%)"); ax.set_ylabel("transition windows caught (%)"); ax.legend(fontsize=8.5, loc="lower right"); save(fig, "tradeoff")

# 12. feature importance
imp = sorted(CL["rf_importance"].items(), key=lambda x: x[1]); lab = {"p_rms": "RMS of p′", "kurtosis": "kurtosis", "peak_ratio": "FFT peak ratio", "phase_lag": "phase lag Δθ", "zcr_reg": "zero-crossing regularity"}
fig, ax = plt.subplots(figsize=(4.2, 2.8)); ax.barh([lab[k] for k, _ in imp], [v * 100 for _, v in imp], color="0.45")
for i, (k, v) in enumerate(imp): ax.text(v * 100 + 0.5, i, f"{v * 100:.1f}%", va="center", fontsize=8.5)
ax.set_xlim(0, 45); ax.set_xlabel("mean decrease in impurity (%)"); ax.grid(axis="y", alpha=0); save(fig, "importance")

# 13. training curves
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for k, c in [("cnn", "#9467bd"), ("lstm_p", "#d62728"), ("lstm_pq", "#2ca02c")]:
    h = json.load(open(R + f"{k}_train.json"))["history"]; ax.plot([p["step"] for p in h], [p["val_acc"] * 100 for p in h], "-o", ms=3, color=c, label=SHORT[k])
ax.set_xlabel("training step"); ax.set_ylabel("validation accuracy (%)"); ax.legend(fontsize=9); save(fig, "training")

# 14. one fast lean-out: decisions over time
rid = 101; d = load_run(rid); t = d["t_end"]
fig, axs = plt.subplots(4, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 0.5, 0.5, 0.5]})
axs[0].plot(np.arange(len(d["p"])) / 1e4, d["p"], color="0.25", lw=0.4); axs[0].set_ylabel("p′ (kPa)"); axs[0].set_ylim(-4, 4)
on = M[rid]["onset_idx"]; [a.axvline(t[on], color="k", ls="--", lw=1) for a in axs]
axs[0].text(t[on] + 0.03, 3.2, "full instability", fontsize=8)
def lane(ax, y, title):
    for c in range(3):
        m = y == c; ax.scatter(t[m], np.zeros(m.sum()), marker="|", s=120, color=CC[c], linewidths=1.2)
    ax.set_yticks([]); ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8.5); ax.grid(False)
lane(axs[1], d["y"], "true regime")
from train_classical import rms_rule
lane(axs[2], rms_rule(d["F"]), "RMS 3-level")
z = np.load(R + "probs/lstm_pq.npz"); lane(axs[3], decide(z[str(rid)], SUM["tau"]["lstm_pq"]), "CNN-LSTM (p′+q′)")
for ax, pr in [(axs[2], rms_rule(d["F"])), (axs[3], decide(z[str(rid)], SUM["tau"]["lstm_pq"]))]:
    a = alarm_track(pr); ax.fill_between(t, -0.9, -0.6, where=a, color=C2, step="mid", lw=0); ax.set_ylim(-1, 1)
axs[-1].set_xlabel("time (s)"); fig.subplots_adjust(hspace=0.15); save(fig, "replay101")

# 15. real recordings examples
ex = json.load(open(R + "real_examples.json"))
fig, axs = plt.subplots(1, 3, figsize=(9, 2.3), sharey=False)
for ax, (k, n), c in zip(axs, [("Stable", "stable"), ("Transient", "transient"), ("LimitCycle", "limit cycle")], CC):
    v = np.array(ex[k][:500]); ax.plot(np.arange(len(v)) / 5, v, color=c, lw=0.7); ax.set_title(n, fontsize=10); ax.set_xlabel("time (ms)")
axs[0].set_ylabel("pressure (rig units)"); fig.tight_layout(); save(fig, "realdata")
print("figures done")
