"""
Real-data check on public combustor pressure recordings
(github.com/Hashim-mirjan/combustion-stability-classifier-mech0020, folder TestingData/):
14 Stable + 14 Transient + 14 Limit-cycle recordings, 0.5 s each at 10 kHz, pressure only.

A) In-domain: Random Forest on the 4 pressure-only features, grouped 7-fold CV by recording.
B) Zero-shot transfer: RF trained only on our simulated runs with rig-independent features
   (RMS normalised by the rig's stable baseline, kurtosis, peak ratio, zero-crossing regularity),
   applied unchanged to the real recordings (report Section 7.4).
"""
import json, sys
import numpy as np
import openpyxl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from features import make_windows, window_features
from common import load_run, run_ids, RES

SRC = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).parent / "real_data_TestingData")
CLS = ["Stable", "Transient", "LimitCycle"]


def load_real():
    recs = []
    for c, name in enumerate(CLS):
        wb = openpyxl.load_workbook(f"{SRC}/{name}.xlsx", read_only=True)
        X = np.array([r[1:] for r in wb.active.iter_rows(values_only=True)], dtype=float)
        X = X[:, ~np.isnan(X).any(0)]
        for j in range(X.shape[1]):
            recs.append((c, j, X[:, j].astype(np.float32)))
    return recs


def cm_of(y, p):
    cm = np.zeros((3, 3), int); np.add.at(cm, (y, p), 1); return cm


if __name__ == "__main__":
    recs = load_real()
    Xs, ys, gs, fpk_all = [], [], [], []
    for gi, (c, j, x) in enumerate(recs):
        W, _ = make_windows(x)
        F, fpk = window_features(W)
        Xs.append(F[:, :4]); ys.append(np.full(len(F), c)); gs.append(np.full(len(F), gi)); fpk_all.append(fpk)
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    print("real windows", X.shape, np.bincount(y))

    # per-class signal summary for the website
    summ = []
    for c in range(3):
        m = y == c
        summ.append(dict(cls=CLS[c], n_rec=14, rms_med=float(np.median(X[m, 0])), kurt_med=float(np.median(X[m, 1])),
                         peak_ratio_med=float(np.median(X[m, 2])),
                         f_peak_med=float(np.median(np.concatenate([f for (cc, _, _), f in zip(recs, fpk_all) if cc == c])))))

    # A) in-domain grouped CV
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(7).split(X, y, g):
        m = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    cmA = cm_of(y, pred); accA = float((pred == y).mean())
    # recording-level majority vote
    rec_pred = np.array([np.bincount(pred[g == i], minlength=3).argmax() for i in range(len(recs))])
    rec_true = np.array([r[0] for r in recs])
    rec_acc = float((rec_pred == rec_true).mean())
    rf_full = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, y)
    impA = rf_full.feature_importances_.tolist()
    print("A in-domain window acc %.3f, recording acc %.3f" % (accA, rec_acc)); print(cmA)

    # RMS 3-level rule as-is (assumes kPa) for reference
    p_rule = np.where(X[:, 0] < 0.3, 0, np.where(X[:, 0] < 1.2, 1, 2))
    acc_rule = float((p_rule == y).mean()); cm_rule = cm_of(y, p_rule)
    print("RMS rule on real: %.3f" % acc_rule)

    # B) zero-shot transfer from simulation
    def rig_independent(F, baseline):
        Z = F[:, :4].copy(); Z[:, 0] = Z[:, 0] / baseline; return Z
    Xs_sim, ys_sim = [], []
    for r in run_ids("train"):
        d = load_run(r)
        base = np.median(d["F"][:200, 0])  # first 2 s of each run = known-stable start-up
        Xs_sim.append(rig_independent(d["F"], base)); ys_sim.append(d["y"])
    rf_sim = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(
        np.concatenate(Xs_sim), np.concatenate(ys_sim))
    base_real = np.median(X[y == 0, 0])  # rig baseline from its stable recordings
    pB = rf_sim.predict(rig_independent(X, base_real))
    cmB = cm_of(y, pB); accB = float((pB == y).mean())
    print("B zero-shot sim->real acc %.3f" % accB); print(cmB)

    json.dump(dict(source="github.com/Hashim-mirjan/combustion-stability-classifier-mech0020 (TestingData/)",
                   n_recordings=len(recs), n_windows=int(len(y)), class_summary=summ,
                   in_domain=dict(acc=accA, recording_acc=rec_acc, confusion=cmA.tolist(), importance=impA),
                   rms_rule=dict(acc=acc_rule, confusion=cm_rule.tolist()),
                   zero_shot=dict(acc=accB, confusion=cmB.tolist(), baseline_real=float(base_real))),
              open(RES / "real_check.json", "w"), indent=1)

    # example traces for the website (first recording of each class, 0.25 s, decimated x2)
    ex = {CLS[c]: next(x for (cc, _, x) in recs if cc == c)[:2500:2].round(3).tolist() for c in range(3)}
    json.dump(ex, open(RES / "real_examples.json", "w"))
