"""Generate LaTeX table fragments directly from the result files (no hand-typed numbers)."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root (report/scripts/ -> repo)
REPORT = Path(__file__).resolve().parents[1]

import json, sys
import numpy as np
sys.path.insert(0, str(REPO))
from common import meta
R = str(REPO / "results") + "/"; T = str(REPORT / "tables") + "/"
import os; os.makedirs(T, exist_ok=True)
S = json.load(open(R + "summary.json")); CL = json.load(open(R + "classical.json")); RC = json.load(open(R + "real_check.json"))
runs = meta()["runs"]
MN = ["RMS danger alarm (1.2 kPa)", "RMS 3-level threshold", "SVM (5 features)", "Random Forest (5 features)", "Random Forest + 100 ms smoothing", "1D-CNN (raw p′)", "CNN-LSTM (raw p′)", "CNN-LSTM (p′ + q′ fusion)"]
TEX = {"RMS danger alarm (1.2 kPa)": "RMS danger alarm (1.2\\,kPa)", "RMS 3-level threshold": "RMS 3-level rule", "SVM (5 features)": "SVM (5 features)",
       "Random Forest (5 features)": "Random Forest (5 features)", "Random Forest + 100 ms smoothing": "RF + 100\\,ms smoothing", "1D-CNN (raw p′)": "1D-CNN (raw $p'$)",
       "CNN-LSTM (raw p′)": "CNN-LSTM ($p'$)", "CNN-LSTM (p′ + q′ fusion)": "CNN-LSTM ($p'+q'$)"}
def ms(v):
    if not np.isfinite(v): return "--"
    out = f"{abs(v)/1000:.2f}\\,s" if abs(v) >= 1000 else f"{abs(v):.0f}\\,ms"
    return ("$-$" if v < 0 else "") + out
def pct(v, d=1): return f"{v*100:.{d}f}"
def lat(v): return "$<$0.1" if v is None or v < 0.1 else f"{v:.2f}"

def main_table(scn, mode, fname, caption, label):
    rows = {r["model"]: r for r in S[mode][scn]}
    L = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrr}\\toprule",
         "Detector & Acc. (\\%) & Macro-F1 & Trans. caught (\\%) & False alarms (\\%) & Median warning & Worst run & Runs $\\geq$40\\,ms \\\\\\midrule"]
    for m in MN:
        r = rows[m]; n = len(r["per_run"]); bold = m.startswith("CNN-LSTM (p′ + q′")
        cells = [TEX[m], pct(r["accuracy"]), f"{r['macro_f1']:.3f}", pct(r["recall"][1]), pct(r["false_alarm_rate"]), ms(r["lead_ms_median"]), ms(r["lead_ms_min"]), f"{round(r['pct_runs_lead_ge_valve']*n)}/{n}"]
        if bold: cells = ["\\textbf{" + c + "}" for c in cells]
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule\\end{tabular}}", "\\end{table}"]
    open(T + fname, "w").write("\n".join(L))

main_table("fast", "tuned", "main_fast.tex", "Results on the 12 unseen fast lean-out runs. Thresholds of the probabilistic detectors were tuned on the validation runs for at most 5\\% false alarms.", "tab:fast")
main_table("test", "tuned", "main_slow.tex", "Results on the two held-out slow sweeps (30\\,s each), same operating point as Table~\\ref{tab:fast}.", "tab:slow")
main_table("fast", "default", "default_fast.tex", "Fast lean-outs with plain arg-max decisions (no threshold tuning). False alarms are much higher, which also inflates warning time.", "tab:defaultfast")
main_table("test", "default", "default_slow.tex", "Slow sweeps with plain arg-max decisions.", "tab:defaultslow")

# latency + per-class
rows = {r["model"]: r for r in S["tuned"]["fast"]}
L = ["\\begin{table}[H]\\centering\\small", "\\caption{Per-class recall on the fast lean-outs and computation time per decision (one CPU core, feature extraction included).}\\label{tab:latency}",
     "\\begin{tabular}{lrrrr}\\toprule", "Detector & Stable (\\%) & Transition (\\%) & Unstable (\\%) & Time/decision (ms) \\\\\\midrule"]
for m in MN:
    r = rows[m]; L.append(" & ".join([TEX[m], pct(r["recall"][0]), pct(r["recall"][1]), pct(r["recall"][2]), lat(r.get("latency_ms"))]) + " \\\\")
L += ["\\bottomrule\\end{tabular}", "\\end{table}"]; open(T + "latency.tex", "w").write("\n".join(L))

# quiet gap
Q = S["quiet_gap"]
L = ["\\begin{table}[H]\\centering\\small", "\\caption{Recall inside the true transition, split into windows containing a burst (RMS $\\geq$ 0.3\\,kPa) and windows in a calm gap (RMS $<$ 0.3\\,kPa), slow sweeps. \\textit{Flips} counts how often the decision switches between stable and non-stable per second of transition.}\\label{tab:gap}",
     "\\begin{tabular}{lrrr}\\toprule", "Detector & Burst windows (\\%) & Calm-gap windows (\\%) & Flips per second \\\\\\midrule"]
for m in MN:
    if m in Q: q = Q[m]; L.append(" & ".join([TEX[m], pct(q["burst_recall"], 0), pct(q["gap_recall"], 0), f"{q['flips_per_s']:.1f}"]) + " \\\\")
L += ["\\bottomrule\\end{tabular}", "\\end{table}"]; open(T + "gap.tex", "w").write("\n".join(L))
gapfrac = list(Q.values())[0]["gap_fraction"]

# real data
cm = {k: np.array(RC[k]["confusion"]) for k in ["in_domain", "zero_shot", "rms_rule"]}
L = ["\\begin{table}[H]\\centering\\small", "\\caption{Window accuracy on the 42 public combustor recordings (1{,}932 windows of 50\\,ms), overall and per true class.}\\label{tab:real}",
     "\\begin{tabular}{lrrrr}\\toprule", "Model & Overall (\\%) & Stable (\\%) & Transient (\\%) & Limit cyc. (\\%) \\\\\\midrule"]
for k, n in [("in_domain", "RF trained on recordings (grouped CV)"), ("zero_shot", "RF trained on simulation (zero-shot)"), ("rms_rule", "RMS 3-level rule")]:
    pc = np.diag(cm[k]) / cm[k].sum(1); L.append(" & ".join([n, pct(RC[k]["acc"]), *[pct(v) for v in pc]]) + " \\\\")
L += ["\\bottomrule\\end{tabular}", "\\end{table}"]; open(T + "real.tex", "w").write("\n".join(L))

# window study
L = ["\\begin{table}[H]\\centering\\small", "\\caption{Effect of window length on the Random Forest (plain arg-max decisions).}\\label{tab:window}",
     "\\begin{tabular}{rrrrr}\\toprule", "Window & Acc. slow (\\%) & False alarms slow (\\%) & Acc. fast (\\%) & Median warning fast \\\\\\midrule"]
for w in CL["window_study"]:
    L.append(" & ".join([f"{w['window_ms']:.0f}\\,ms", pct(w["test_acc"]), pct(w["test_far"]), pct(w["fast_acc"]), ms(w["fast_lead_med"])]) + " \\\\")
L += ["\\bottomrule\\end{tabular}", "\\end{table}"]; open(T + "window.tex", "w").write("\n".join(L))

# deep models
L = ["\\begin{table}[H]\\centering\\small", "\\caption{Deep models: size, training cost and best validation accuracy (plain arg-max).}\\label{tab:deep}",
     "\\begin{tabular}{lrrrr}\\toprule", "Model & Parameters & Training time & Best val. acc. (\\%) & Time per decision (ms) \\\\\\midrule"]
for k, n in [("cnn", "1D-CNN (raw $p'$)"), ("lstm_p", "CNN-LSTM ($p'$)"), ("lstm_pq", "CNN-LSTM ($p'+q'$)")]:
    j = json.load(open(R + f"{k}_train.json")); L.append(f"{n} & " + f"{j['n_params']:,}".replace(",", "{,}") + f" & {j['train_time_s']/60:.1f}\\,min & {pct(j['best_val_acc'])} & {j['latency_ms']:.2f} \\\\")
L += ["\\bottomrule\\end{tabular}", "\\end{table}"]; open(T + "deep.tex", "w").write("\n".join(L))

# dataset per-run table (ramps)
ramps = [r for r in runs if r["kind"] == "ramp"]; fast = [r for r in runs if r["kind"] == "fast"]
sp = meta()["split"]; role = {**{i: "train" for i in sp["train"]}, **{i: "val" for i in sp["val"]}, **{i: "test" for i in sp["test"]}}
L = ["\\begin{table}[H]\\centering\\footnotesize", "\\caption{The 15 slow sweeps: run-to-run variation, regime boundaries and number of windows per class.}\\label{tab:runs}",
     "\\begin{tabular}{rlrrrrrrr}\\toprule", "Run & Split & Flow factor & $f_\\mathrm{ref}$ (Hz) & $\\phi_{S\\to T}$ & $\\phi_{T\\to U}$ & Stable & Transition & Unstable \\\\\\midrule"]
for r in ramps:
    c = r["class_counts"]; L.append(f"{r['run_id']} & {role[r['run_id']]} & {r['flow_factor']:.3f} & {r['f_ref']:.1f} & {r['phi_ST']:.3f} & {r['phi_TU']:.3f} & {c[0]} & {c[1]} & {c[2]} \\\\")
tot = np.sum([r["class_counts"] for r in ramps], 0)
L += ["\\midrule", (f"\\multicolumn{{6}}{{l}}{{Total ({sum(r['n_windows'] for r in ramps):,} windows)}} & {tot[0]:,} & {tot[1]:,} & {tot[2]:,}".replace(",", "{,}") + " \\\\"), "\\bottomrule\\end{tabular}", "\\end{table}"]
open(T + "runs.tex", "w").write("\n".join(L))

facts = dict(n_windows=int(sum(r["n_windows"] for r in ramps)), tot=tot.tolist(), gapfrac=gapfrac, tau=S["tau"],
             fast_dur=[min(r["duration"] for r in fast), max(r["duration"] for r in fast)], lat=CL["latency"], imp=CL["rf_importance"],
             phiST=[min(r["phi_ST"] for r in ramps), max(r["phi_ST"] for r in ramps)], phiTU=[min(r["phi_TU"] for r in ramps), max(r["phi_TU"] for r in ramps)])
json.dump(facts, open(T + "facts.json", "w"), indent=1); print(json.dumps(facts, indent=1))
