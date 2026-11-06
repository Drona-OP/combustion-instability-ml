"""Shared loading + evaluation (alarm logic, lead time, false alarms)."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CLASSES = ["Stable", "Transition", "Unstable"]
STRIDE_S = 0.010
ALARM_ON = 3      # consecutive windows predicted >= Transition to raise the alarm (report 7.1)
ALARM_OFF = 10    # consecutive Stable predictions to clear it
VALVE_MS = 40.0   # actuator response time quoted in the report


def meta():
    return json.load(open(DATA / "meta.json"))


def load_run(rid):
    return dict(np.load(DATA / f"run_{rid:03d}.npz"))


def run_ids(split):
    m = meta()
    if split == "fast":
        return [r["run_id"] for r in m["runs"] if r["kind"] == "fast"]
    return m["split"][split]


def alarm_track(pred):
    """Hysteretic alarm: ON after ALARM_ON consecutive non-stable predictions, OFF after ALARM_OFF stable."""
    on = np.zeros(len(pred), bool)
    state, cnt_hi, cnt_lo = False, 0, 0
    for i, c in enumerate(pred):
        if c >= 1:
            cnt_hi += 1; cnt_lo = 0
        else:
            cnt_lo += 1; cnt_hi = 0
        if not state and cnt_hi >= ALARM_ON:
            state = True
        elif state and cnt_lo >= ALARM_OFF:
            state = False
        on[i] = state
    return on


def run_events(pred, y, onset_idx):
    """Lead time (ms) of the alarm episode active at instability onset + false-alarm episodes."""
    on = alarm_track(pred)
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    stable_idx = np.flatnonzero(y == 0)
    # false alarm episodes: alarm raised while the rig is truly Stable
    false_eps = int(sum(1 for s in starts if y[s] == 0))
    stable_minutes = len(stable_idx) * STRIDE_S / 60.0
    trans_start = int(np.argmax(y >= 1)) if (y >= 1).any() else 0
    if onset_idx < 0:
        lead = np.nan
    elif on[onset_idx]:
        s = starts[starts <= onset_idx].max()
        s = max(s, trans_start)  # warning only counts once the rig is truly in transition
        lead = (onset_idx - s) * STRIDE_S * 1000.0
    else:
        after = starts[starts > onset_idx]
        lead = -((after.min() - onset_idx) * STRIDE_S * 1000.0) if len(after) else -np.inf
    # earliest *valid* warning: first alarm start that is not a false alarm, before onset
    avail_ms = (onset_idx - trans_start) * STRIDE_S * 1000.0 if onset_idx >= 0 else np.nan
    return dict(lead_ms=float(lead), false_episodes=false_eps, stable_minutes=stable_minutes, alarm=on,
                avail_ms=float(avail_ms))


def summarize(name, preds, ys, onsets, latency_ms=None, extra=None):
    """preds/ys/onsets: dict run_id -> arrays/int."""
    P = np.concatenate([preds[r] for r in preds]); Y = np.concatenate([ys[r] for r in preds])
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (Y, P), 1)
    acc = float((P == Y).mean())
    recall = (np.diag(cm) / cm.sum(1).clip(1)).tolist()
    prec = (np.diag(cm) / cm.sum(0).clip(1)).tolist()
    f1 = [2 * p * r / (p + r) if p + r > 0 else 0.0 for p, r in zip(prec, recall)]
    far = float(cm[0, 1:].sum() / cm[0].sum())  # stable windows flagged as non-stable
    leads, fe, sm = [], 0, 0.0
    per_run = {}
    for r in preds:
        ev = run_events(preds[r], ys[r], onsets[r])
        leads.append(ev["lead_ms"]); fe += ev["false_episodes"]; sm += ev["stable_minutes"]
        per_run[int(r)] = dict(lead_ms=ev["lead_ms"], false_episodes=ev["false_episodes"], avail_ms=ev["avail_ms"])
    leads = np.array(leads)
    out = dict(model=name, accuracy=acc, macro_f1=float(np.mean(f1)), recall=recall, precision=prec, f1=f1,
               false_alarm_rate=far, confusion=cm.tolist(),
               lead_ms_median=float(np.median(leads)), lead_ms_min=float(np.min(leads)),
               lead_ms_mean=float(np.mean(leads[np.isfinite(leads)])) if np.isfinite(leads).any() else None,
               pct_runs_lead_ge_valve=float(np.mean(leads >= VALVE_MS)),
               false_alarm_episodes=int(fe), false_alarms_per_min=float(fe / max(sm, 1e-9)),
               per_run=per_run)
    if latency_ms is not None:
        out["latency_ms"] = latency_ms
    if extra:
        out.update(extra)
    return out


def decide(proba, tau):
    """Raise non-stable only if P(Transition)+P(Unstable) > tau; then pick the likelier of T/U."""
    ns = proba[:, 1] + proba[:, 2]
    tu = np.where(proba[:, 2] > proba[:, 1], 2, 1)
    return np.where(ns > tau, tu, 0)


def tune_tau(probas, ys, max_far=0.05):
    """Smallest tau giving window-level false-alarm rate <= max_far on validation runs."""
    P = np.concatenate(probas); Y = np.concatenate(ys)
    ns = P[:, 1] + P[:, 2]
    st = ns[Y == 0]
    for tau in np.linspace(0.3, 0.995, 140):
        if (st > tau).mean() <= max_far:
            return float(tau)
    return 0.995
