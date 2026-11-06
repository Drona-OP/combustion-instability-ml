// In-browser port of simulator.py + features.py + Random Forest inference.
// Same equations and constants as the Python code; used by the live simulator tab.
(function (root) {
  const FS = 10000, DT = 1 / FS, WIN = 500, NFFT = 2048, WG_DELAY = 0.5e-3;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussFactory(seed) {
    const u = mulberry32(seed); let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let a, b, r;
      do { a = 2 * u() - 1; b = 2 * u() - 1; r = a * a + b * b; } while (r === 0 || r >= 1);
      const m = Math.sqrt(-2 * Math.log(r) / r); spare = b * m; return a * m;
    };
  }

  const DEFAULTS = {
    flow_factor: 1.0, alpha: 60.4, beta: 92.6, theta0_deg: 30.0, dtheta_dphi: 214.3,
    c3: 5.33, c5: 0.889, sigma: 2.2, s_nu: 25.0, tau_nu: 0.10, a_lock: 1.0,
    f_ref: 210.0, turb_rms: 0.07, sensor_rms: 0.015, flame_noise: 0.35, h2: 0.06,
  };

  class Combustor {
    constructor(params = {}, seed = 12345) {
      this.P = Object.assign({}, DEFAULTS, params);
      this.g = gaussFactory(seed);
      this.zr = 0.05; this.zi = 0; this.eta = 0; this.Phi = 0;
      this.turb = 0; this.flame = 0; this.m2a = 0; this.m2b = 0;
      this.delay = new Float64Array(8); this.di = 0;
      this.dw = 2 * Math.PI * 1.5 * this.g();
      this.phi = 0.9;
    }
    thetaDeg(phi) { const P = this.P; return (P.theta0_deg + P.dtheta_dphi * (phi - 0.55)) / P.flow_factor; }
    nu(phi) { return this.P.beta * Math.cos(this.thetaDeg(phi) * Math.PI / 180) - this.P.alpha; }
    nuT() { return -2 * this.P.s_nu; }
    regime(phi) { const n = this.nu(phi); return n < this.nuT() ? 0 : (n < 0 ? 1 : 2); }
    freq(phi) { return this.P.f_ref + 22.0 * (phi - 0.60); }
    // generate n samples into p, q (Float32Array) at the current phi
    run(n, p, q, off = 0) {
      const P = this.P, g = this.g, phi = this.phi;
      const nu = this.nu(phi), th = this.thetaDeg(phi) * Math.PI / 180, f = this.freq(phi);
      const rho = Math.exp(-DT / P.tau_nu), kr = Math.sqrt(1 - rho * rho) * P.s_nu;
      const ns = P.sigma * Math.sqrt(DT) / Math.SQRT2, inv = 1 / (P.a_lock * P.a_lock);
      const a1 = Math.exp(-2 * Math.PI * 120 / FS), s1 = 1 / Math.sqrt((1 - a1) / (1 + a1));
      const a2 = Math.exp(-2 * Math.PI * 400 / FS), s2 = 1 / Math.sqrt((1 - a2) / (1 + a2));
      const r = Math.exp(-Math.PI * 40 / FS), w2 = 2 * Math.PI * 3.05 * P.f_ref / FS, c = Math.cos(w2);
      const var2 = (1 + r * r) / ((1 - r * r) * ((1 + r * r) ** 2 - 4 * r * r * c * c)), s3 = 1 / Math.sqrt(var2);
      const flowScale = P.flow_factor * (1 + 0.6 * (0.90 - phi) / 0.35);
      const dPhi = 2 * Math.PI * f * DT, cth = Math.cos(th), sth = Math.sin(th);
      for (let k = 0; k < n; k++) {
        this.eta = rho * this.eta + kr * g();
        const r2 = this.zr * this.zr + this.zi * this.zi;
        const gr = nu + this.eta / (1 + r2 * inv) + P.c3 * r2 - P.c5 * r2 * r2;
        const zr = this.zr, zi = this.zi;
        this.zr = zr + (gr * zr - this.dw * zi) * DT + ns * g();
        this.zi = zi + (gr * zi + this.dw * zr) * DT + ns * g();
        this.Phi += dPhi; if (this.Phi > 6.283185307179586) this.Phi -= 6.283185307179586;
        const cr = Math.cos(this.Phi), ci = Math.sin(this.Phi);
        const xr = this.zr * cr - this.zi * ci, xi = this.zr * ci + this.zi * cr; // a e^{i Phi}
        const pm = xr + P.h2 * (xr * xr - xi * xi);
        this.turb = (1 - a1) * g() + a1 * this.turb;
        const m2 = g() + 2 * r * c * this.m2a - r * r * this.m2b; this.m2b = this.m2a; this.m2a = m2;
        const pt = pm + P.turb_rms * flowScale * this.turb * s1 + 0.035 * m2 * s3;
        // waveguide delay (5 samples) + attenuation + sensor noise
        const d = this.delay; const out = d[this.di]; d[this.di] = pt; this.di = (this.di + 1) % 5;
        p[off + k] = 0.97 * out + P.sensor_rms * g();
        this.flame = (1 - a2) * g() + a2 * this.flame;
        const qc = xr * cth + xi * sth; // Re{a e^{i(Phi - theta)}}
        q[off + k] = qc + P.flame_noise * this.flame * s2 + 0.02 * g();
      }
      return { nu, theta: th * 180 / Math.PI, f, amp: Math.hypot(this.zr, this.zi) };
    }
  }

  // ---------------- FFT (radix-2, in place) ----------------
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = i + j, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }
  const HANN = new Float64Array(WIN).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (WIN - 1)));

  // features of one window; pw/qw: arrays of length 500. Returns {F:[5], fpk, psd(0..2kHz)}
  function features(pw, qw) {
    const n = pw.length; let mp = 0, mq = 0;
    for (let i = 0; i < n; i++) { mp += pw[i]; mq += qw[i]; }
    mp /= n; mq /= n;
    const pr = new Float64Array(NFFT), pi = new Float64Array(NFFT), qr = new Float64Array(NFFT), qi = new Float64Array(NFFT);
    let s2 = 0, s4 = 0, zc = 0, prev = pw[0] - mp < 0;
    for (let i = 0; i < n; i++) {
      const x = pw[i] - mp; s2 += x * x; s4 += x * x * x * x;
      const neg = x < 0; if (i > 0 && neg !== prev) zc++; prev = neg;
      pr[i] = x * HANN[i]; qr[i] = (qw[i] - mq) * HANN[i];
    }
    const v = s2 / n, rms = Math.sqrt(v), kurt = (s4 / n) / (v * v + 1e-20);
    fft(pr, pi); fft(qr, qi);
    const df = FS / NFFT, nb = NFFT / 2 + 1;
    let kpk = -1, best = -1;
    for (let k = 0; k < nb; k++) { const f = k * df; if (f >= 80 && f <= 2000) { const P = pr[k] * pr[k] + pi[k] * pi[k]; if (P > best) { best = P; kpk = k; } } }
    const fpk = kpk * df;
    let inb = 0, tot = 0, cr = 0, ci = 0;
    for (let k = 0; k < nb; k++) {
      const f = k * df, P = pr[k] * pr[k] + pi[k] * pi[k];
      if (f >= 20 && f <= 2000) tot += P;
      if (Math.abs(f - fpk) <= 30) {
        inb += P;
        cr += pr[k] * qr[k] + pi[k] * qi[k]; ci += pi[k] * qr[k] - pr[k] * qi[k]; // FP * conj(FQ)
      }
    }
    let ang = Math.atan2(ci, cr) - 2 * Math.PI * fpk * WG_DELAY;
    ang = ((ang + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    const zcr = zc * FS / (n - 1);
    const psd = new Float32Array(Math.floor(1000 / df));
    for (let k = 0; k < psd.length; k++) psd[k] = pr[k] * pr[k] + pi[k] * pi[k];
    return { F: [rms, kurt, inb / (tot + 1e-20), zcr / (2 * fpk), Math.abs(ang) * 180 / Math.PI], fpk, psd, df };
  }

  // ---------------- Random forest ----------------
  function b64(s, T) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new T(u.buffer); }
  class Forest {
    constructor(j) {
      this.feat = b64(j.feat, Int8Array); this.thr = b64(j.thr, Float32Array);
      this.left = b64(j.left, Int32Array); this.right = b64(j.right, Int32Array);
      this.val = b64(j.val, Uint8Array); this.roots = j.roots; this.tau = j.tau;
    }
    proba(F) {
      const x = F.map(Math.fround), out = [0, 0, 0];
      for (const r of this.roots) {
        let k = r;
        while (this.feat[k] >= 0) k = (x[this.feat[k]] <= this.thr[k]) ? this.left[k] : this.right[k];
        out[0] += this.val[3 * k]; out[1] += this.val[3 * k + 1]; out[2] += this.val[3 * k + 2];
      }
      const s = out[0] + out[1] + out[2]; return out.map(v => v / s);
    }
    decide(p) { const ns = p[1] + p[2]; return ns > this.tau ? (p[2] > p[1] ? 2 : 1) : 0; }
  }

  root.CI = { Combustor, features, Forest, FS, WIN, DEFAULTS, fft };
})(typeof window !== "undefined" ? window : globalThis);
