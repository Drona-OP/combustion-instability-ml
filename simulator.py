"""
Reduced-order thermoacoustic combustor simulator
================================================

Generates synchronised acoustic-pressure (p') and heat-release (q', photodiode)
signals at 10 kHz for a lean-premixed swirl combustor whose equivalence ratio
phi is ramped down, so the rig passes Stable -> Intermittency -> Limit cycle.

Physics (one acoustic mode, stochastic slow-flow / Noiray-type oscillator):

    da/dt = (nu + i*dw) a + c3 |a|^2 a - c5 |a|^4 a + sigma * xi(t)

    p'(t) = Re{ a(t) exp(i*Phi(t)) } + harmonics + 2nd mode + turbulence + sensor noise
    q'(t) = n * Re{ a(t) exp(i*(Phi(t) - theta)) } + flame noise          (n-tau flame)

  * a(t)  : complex acoustic amplitude of the 1/4-wave mode (kPa)
  * nu    : linear growth rate = flame driving - acoustic damping
            nu(phi) = beta * cos(theta(phi)) - alpha      <- Rayleigh criterion
            The flame only adds energy when |theta| < 90 deg (cos > 0), and the
            mode only grows when that driving beats the damping alpha.
  * theta : phase lag of heat release behind pressure, theta = omega * tau,
            tau = convective flame delay ~ 1 / bulk velocity. Lowering phi by
            adding air shortens tau and slides theta from ~105 deg to ~30 deg.
  * c3 > 0, c5 > 0 : subcritical (bistable) Hopf bifurcation, as observed in
            lean-premixed combustors. For nu_sn < nu < 0 both the quiet state
            and the limit cycle are stable, and turbulent noise kicks the
            system between them -> intermittent bursts separated by calm gaps.

  * eta(t): turbulent fluctuation of the flame gain (OU process, std s_nu,
            ~0.1 s correlation). It is suppressed once the flame locks on to the
            acoustic wave (|a| > a_lock), as seen in real limit cycles.

Ground-truth regime (known exactly in simulation, per run), from the mean growth rate:
    Stable      : nu < -2 s_nu     flame driving essentially never beats damping
    Transition  : -2 s_nu <= nu < 0  turbulent gain fluctuations intermittently push
                                   the net growth positive -> bursts + calm gaps
    Unstable    : nu >= 0          mean driving > damping -> self-sustained limit cycle
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict

FS = 10_000  # Hz
DT = 1.0 / FS
WAVEGUIDE_DELAY_S = 0.5e-3  # semi-infinite cooling tube between wall and transducer


@dataclass
class RunParams:
    run_id: int
    seed: int
    kind: str = "ramp"             # "ramp" (30 s slow sweep) or "fast" (lean-out transient)
    duration: float = 30.0
    phi_start: float = 0.90
    phi_end: float = 0.55
    hold_start: float = 0.0        # seconds held at phi_start before ramp
    hold_end: float = 0.0          # seconds held at phi_end after ramp
    flow_factor: float = 1.0       # bulk velocity / reference (shifts tau -> regime boundaries)
    alpha: float = 60.4            # acoustic damping (1/s)
    beta: float = 92.6             # flame driving gain (1/s)
    theta0_deg: float = 30.0       # flame phase lag at phi = 0.55, reference flow
    dtheta_dphi: float = 214.3     # deg per unit phi
    c3: float = 5.33               # 1/(s kPa^2)   subcritical (bursts grow fast once started)
    c5: float = 0.889              # 1/(s kPa^4)   saturation -> limit-cycle amplitude ~2.4 kPa
    sigma: float = 2.2             # additive acoustic forcing by turbulent combustion noise
    s_nu: float = 25.0             # multiplicative noise: turbulent fluctuation of flame gain (1/s)
    tau_nu: float = 0.10           # correlation time of flame-gain fluctuations (s)
    a_lock: float = 1.0            # kPa: above this the flame locks to the acoustics and
                                   # turbulent gain fluctuations are suppressed
    f_ref: float = 210.0           # Hz, quarter-wave mode at phi ~ 0.6
    turb_rms: float = 0.07         # kPa, broadband turbulent rumble at reference flow
    sensor_rms: float = 0.015      # kPa
    flame_noise: float = 0.35      # photodiode broadband noise relative to coherent part
    h2: float = 0.06               # 2nd harmonic strength (wave steepening)

    # ---- derived helpers -------------------------------------------------
    def theta_deg(self, phi):
        # tau ~ L / U ; higher flow -> smaller tau -> smaller theta
        return (self.theta0_deg + self.dtheta_dphi * (phi - 0.55)) / self.flow_factor

    def nu(self, phi):
        return self.beta * np.cos(np.deg2rad(self.theta_deg(phi))) - self.alpha

    @property
    def nu_sn(self):
        return -self.c3 ** 2 / (4 * self.c5)

    def freq(self, phi):
        # leaner -> cooler products -> slightly lower sound speed
        return self.f_ref + 22.0 * (phi - 0.60)

    def phi_of_t(self, t):
        ramp = self.duration - self.hold_start - self.hold_end
        s = np.clip((t - self.hold_start) / ramp, 0.0, 1.0)
        return self.phi_start + (self.phi_end - self.phi_start) * s

    @property
    def nu_T(self):
        """Onset of intermittency: mean growth rate within 2 std of turbulent gain
        fluctuations, i.e. the flame intermittently out-drives the damping (P > 2.3%)."""
        return -2.0 * self.s_nu

    def boundaries_phi(self):
        """phi where nu = nu_T (stable->transition) and nu = 0 (transition->unstable)."""
        grid = np.linspace(0.5, 0.95, 20001)
        nu = self.nu(grid)
        p_sn = grid[np.argmin(np.abs(nu - self.nu_T))]
        p_0 = grid[np.argmin(np.abs(nu))]
        return float(p_sn), float(p_0)


def _colored_noise(rng, n, fc, fs=FS):
    """Low-pass (1st order) white noise, unit RMS — turbulent rumble."""
    w = rng.standard_normal(n)
    a = np.exp(-2 * np.pi * fc / fs)
    from scipy.signal import lfilter
    y = lfilter([1 - a], [1, -a], w)
    return y / (y.std() + 1e-12)


def _narrowband(rng, n, f0, bw, fs=FS):
    """Noise-driven damped resonance at f0 with bandwidth bw (unit RMS)."""
    from scipy.signal import lfilter
    r = np.exp(-np.pi * bw / fs)
    th = 2 * np.pi * f0 / fs
    y = lfilter([1.0], [1, -2 * r * np.cos(th), r * r], rng.standard_normal(n))
    return y / (y.std() + 1e-12)


def simulate_run(P: RunParams):
    rng = np.random.default_rng(P.seed)
    n = int(round(P.duration * FS))
    t = np.arange(n) * DT
    phi = P.phi_of_t(t)
    nu = P.nu(phi)
    theta = np.deg2rad(P.theta_deg(phi))
    f = P.freq(phi)

    # --- slow-flow integration (Euler–Maruyama at 10 kHz; slow-flow time scales are ~10-100 ms)
    a = np.empty(n, dtype=np.complex128)
    z = complex(0.05, 0.0)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (P.sigma * np.sqrt(DT) / np.sqrt(2))
    # turbulent flame-gain fluctuation (Ornstein-Uhlenbeck, std s_nu, corr. time tau_nu)
    from scipy.signal import lfilter
    rho = np.exp(-DT / P.tau_nu)
    eta = lfilter([np.sqrt(1 - rho * rho)], [1, -rho], rng.standard_normal(n)) * P.s_nu
    inv_lock2 = 1.0 / P.a_lock ** 2
    dw = 2 * np.pi * 1.5 * rng.standard_normal()  # small detuning (rad/s)
    c3, c5 = P.c3, P.c5
    for k in range(n):
        r2 = z.real * z.real + z.imag * z.imag
        z = z + ((nu[k] + eta[k] / (1.0 + r2 * inv_lock2) + 1j * dw) + c3 * r2 - c5 * r2 * r2) * z * DT + noise[k]
        a[k] = z

    # --- fast carrier
    Phi = 2 * np.pi * np.cumsum(f) * DT
    carrier = np.exp(1j * Phi)
    p_mode = np.real(a * carrier)
    p_h2 = P.h2 * np.real((a * carrier) ** 2) / 1.0  # grows as amplitude^2
    flow_noise_scale = P.flow_factor * (1.0 + 0.6 * (0.90 - phi) / 0.35)  # more air -> louder rumble
    turb = P.turb_rms * flow_noise_scale * _colored_noise(rng, n, fc=120.0)
    mode2 = 0.035 * _narrowband(rng, n, f0=3.05 * P.f_ref, bw=40.0)  # weak 3/4-wave mode
    p_true = p_mode + p_h2 + turb + mode2

    # waveguide: transducer sees the wall pressure delayed by ~0.5 ms, slightly attenuated
    d = int(round(WAVEGUIDE_DELAY_S * FS))
    p_sensor = np.empty_like(p_true)
    p_sensor[d:] = 0.97 * p_true[:-d]
    p_sensor[:d] = 0.97 * p_true[0]
    p_sensor += P.sensor_rms * rng.standard_normal(n)

    # heat release (photodiode): n-tau flame response to the acoustic mode + broadband flame noise
    q_coh = np.real(a * np.exp(1j * (Phi - theta)))
    q = q_coh + P.flame_noise * _colored_noise(rng, n, fc=400.0) + 0.02 * rng.standard_normal(n)

    # ground-truth regime from the bifurcation diagram
    regime = np.where(nu < P.nu_T, 0, np.where(nu < 0, 1, 2)).astype(np.int8)

    return dict(t=t, phi=phi, nu=nu, theta=theta, freq=f, p=p_sensor.astype(np.float32),
                q=q.astype(np.float32), amp=np.abs(a).astype(np.float32), regime=regime,
                params=asdict(P))


# ---------------------------------------------------------------------------
# Experiment design (matches report Section 4.2)
# ---------------------------------------------------------------------------

def ramp_runs(n_runs=15, base_seed=2026):
    """15 slow 30 s sweeps phi 0.90 -> 0.55 with run-to-run rig variation."""
    rng = np.random.default_rng(base_seed)
    runs = []
    for i in range(n_runs):
        runs.append(RunParams(
            run_id=i + 1, seed=int(rng.integers(1e9)), kind="ramp", duration=30.0,
            flow_factor=float(rng.uniform(0.94, 1.06)),
            alpha=float(60.4 * rng.uniform(0.95, 1.05)),
            sigma=float(2.2 * rng.uniform(0.85, 1.15)),
            turb_rms=float(0.07 * rng.uniform(0.8, 1.25)),
            f_ref=float(rng.uniform(205, 215)),
        ))
    return runs


def fast_runs(n_runs=12, base_seed=7):
    """Fast lean-out transients (e.g. fuel-schedule upset): phi 0.86 -> 0.58 in 1.5-4 s."""
    rng = np.random.default_rng(base_seed)
    runs = []
    for i in range(n_runs):
        ramp = float(rng.uniform(1.5, 4.0))
        runs.append(RunParams(
            run_id=100 + i + 1, seed=int(rng.integers(1e9)), kind="fast",
            duration=1.0 + ramp + 1.5, hold_start=1.0, hold_end=1.5,
            phi_start=0.86, phi_end=0.58,
            flow_factor=float(rng.uniform(0.94, 1.06)),
            alpha=float(60.4 * rng.uniform(0.95, 1.05)),
            sigma=float(2.2 * rng.uniform(0.85, 1.15)),
            turb_rms=float(0.07 * rng.uniform(0.8, 1.25)),
            f_ref=float(rng.uniform(205, 215)),
        ))
    return runs


if __name__ == "__main__":
    P = ramp_runs()[0]
    out = simulate_run(P)
    print("boundaries phi (S->T, T->U):", P.boundaries_phi(), "nu_sn", P.nu_sn)
    for c in range(3):
        m = out["regime"] == c
        print(c, "frac", m.mean().round(3), "p rms", out["p"][m].std().round(3))
