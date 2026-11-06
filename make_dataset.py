"""Generate the 15 slow-ramp runs + 12 fast lean-out runs, window them, compute features."""
import json, time
import numpy as np
from pathlib import Path
import simulator as S
from features import make_windows, window_features, WIN, STRIDE

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

SPLIT = {"train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "val": [12, 13], "test": [14, 15]}


def process(P):
    o = S.simulate_run(P)
    Wp, last = make_windows(o["p"])
    Wq, _ = make_windows(o["q"])
    F, fpk = window_features(Wp, Wq)
    y = o["regime"][last]
    t_end = o["t"][last]
    rms = F[:, 0]
    # "full instability" onset (report definition): first time p'rms >= 1.2 kPa and it stays
    # >= 1.2 kPa in >= 90% of the windows over the next 300 ms
    hi = rms >= 1.2
    k = 30
    sustained = np.array([hi[i] and hi[i:i + k].mean() >= 0.9 for i in range(len(hi))])
    onset_idx = int(np.argmax(sustained)) if sustained.any() else -1
    b_sn, b_0 = P.boundaries_phi()
    meta = dict(run_id=P.run_id, kind=P.kind, duration=P.duration, n_windows=int(len(y)),
                onset_idx=onset_idx, onset_t=float(t_end[onset_idx]) if onset_idx >= 0 else None,
                phi_ST=b_sn, phi_TU=b_0, flow_factor=P.flow_factor, f_ref=P.f_ref,
                class_counts=np.bincount(y, minlength=3).tolist())
    np.savez_compressed(OUT / f"run_{P.run_id:03d}.npz", p=o["p"], q=o["q"], phi=o["phi"].astype(np.float32),
                        nu=o["nu"].astype(np.float32), amp=o["amp"], regime=o["regime"],
                        F=F, fpk=fpk.astype(np.float32), y=y, t_end=t_end.astype(np.float32), last=last)
    return meta


if __name__ == "__main__":
    t0 = time.time()
    metas = []
    for P in S.ramp_runs() + S.fast_runs():
        m = process(P)
        metas.append(m)
        print(m["run_id"], m["kind"], m["n_windows"], m["class_counts"], "onset", m["onset_t"],
              "phi bounds %.3f %.3f" % (m["phi_ST"], m["phi_TU"]), f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"runs": metas, "split": SPLIT, "win": WIN, "stride": STRIDE}, open(OUT / "meta.json", "w"), indent=1)
    tot = sum(m["n_windows"] for m in metas if m["kind"] == "ramp")
    print("ramp windows:", tot)
