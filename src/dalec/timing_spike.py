"""
Timing spike: how expensive is one NUTS gradient evaluation?

Throwaway diagnostic. NOT production code, NOT scientifically meaningful.
It reproduces the SHAPE of the real problem — a six-pool DALEC scan of the
right length, with the right number of free parameters, and comparable
arithmetic per step — so the gradient cost is representative.

Run it, read the milliseconds-per-gradient, and use the projection table
to decide calibration block length and whether cluster time is needed.

Usage:  python timing_spike.py
"""

import time
import platform
import numpy as np
import pytensor
import pytensor.tensor as pt

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
STEP_COUNTS = [1460, 3287]      # 4-year block, 9-year block
PARAM_COUNTS = [18, 13]         # pre-screening, post-screening
N_TIMED = 30                    # timed gradient evaluations
N_WARMUP = 3                    # discarded (compilation, cache warming)

rng = np.random.default_rng(0)


def build_gradient_fn(n_steps: int, n_params: int):
    """Compile a function returning the gradient of a scalar loss wrt theta."""
    theta = pt.dvector("theta")

    # Drivers held as constants — matches the real case, where drivers are data.
    T = pt.as_tensor_variable(rng.normal(5.0, 10.0, n_steps))
    I = pt.as_tensor_variable(np.abs(rng.normal(12.0, 8.0, n_steps)) + 0.1)
    doy = pt.as_tensor_variable((np.arange(n_steps) % 365).astype(float))
    obs = pt.as_tensor_variable(rng.normal(0.0, 2.0, n_steps))
    sigma = pt.as_tensor_variable(np.abs(rng.normal(1.0, 0.3, n_steps)) + 0.1)

    f_auto = theta[0]
    f_lab, f_fol, f_roo, f_woo = theta[1], theta[2], theta[3], theta[4]
    th_woo, th_roo, th_lit, th_som, th_min = (theta[5], theta[6], theta[7],
                                              theta[8], theta[9])
    Theta = theta[10]
    c_eff = theta[11]
    c_lma = theta[12]

    def c64(v):
        return pt.as_tensor_variable(np.float64(v))

    if n_params >= 18:
        init = [theta[13], theta[14], theta[15], theta[16], theta[17], c64(10000.0)]
    else:
        init = [c64(v) for v in (100.0, 300.0, 200.0, 8000.0, 150.0, 10000.0)]

    def step(T_t, I_t, doy_t, lab, fol, roo, woo, lit, som):
        # --- ACM-shaped photosynthesis (Chuter/DALEC form, constants inlined) ---
        L = fol / c_lma
        p_N = c_eff * L * pt.exp(0.0111 * T_t)
        g_c = pt.power(2.0, 0.7897) / (0.5 * 6.0 + 0.3783)
        q = 4.22273 - 208.868
        p = p_N / g_c
        disc = pt.sqr(400.0 + q - p) - 4.0 * (400.0 * q - p * 4.22273)
        C_i = 0.5 * (400.0 + q - p + pt.sqrt(pt.maximum(disc, 1e-12)))
        p_D = g_c * (400.0 - C_i)
        E_0 = 7.1929 * pt.sqr(fol) / (pt.sqr(fol) + 2.1001 * pt.sqr(c_lma))
        delta = -0.408 * pt.cos(2.0 * np.pi * ((doy_t + 10.0) % 365.0) / 365.0)
        dl = 24.0 * pt.arccos(
            pt.clip(-pt.tan(np.radians(61.8474)) * pt.tan(delta), -1.0, 1.0)
        ) / np.pi
        p_I = (E_0 * I_t * p_D) / (E_0 * I_t + p_D + 1e-9)
        gpp = pt.maximum(p_I * (0.0156 * dl + 0.0453), 0.0)

        # --- phenology (smooth, correct shape for gradient cost) ---
        s = 365.25 / np.pi
        phi_on = 0.02 * pt.exp(-pt.sqr(pt.sin((doy_t - 120.0) / s) * 1.414 * s / 50.0))
        phi_off = 0.02 * pt.exp(-pt.sqr(pt.sin((doy_t - 250.0) / s) * 1.414 * s / 60.0))

        # --- six pool updates (A1-A6) ---
        tm = pt.exp(Theta * T_t)
        lab_n = (1.0 - phi_on) * lab + f_lab * gpp
        fol_n = (1.0 - phi_off) * fol + phi_on * lab + f_fol * gpp
        roo_n = (1.0 - th_roo) * roo + f_roo * gpp
        woo_n = (1.0 - th_woo) * woo + f_woo * gpp
        lit_n = ((1.0 - (th_lit + th_min) * tm) * lit
                 + th_roo * roo + phi_off * fol)
        som_n = ((1.0 - th_som * tm) * som + th_woo * woo + th_min * tm * lit)

        reco = f_auto * gpp + (th_lit * lit + th_som * som) * tm
        nee = reco - gpp
        return lab_n, fol_n, roo_n, woo_n, lit_n, som_n, nee

    outputs = pytensor.scan(
        fn=step,
        sequences=[T, I, doy],
        outputs_info=init + [None],
        n_steps=n_steps,
        return_updates=False,
    )
    nee = outputs[-1]

    loss = pt.sum(pt.sqr((nee - obs) / sigma))
    grad = pt.grad(loss, theta)
    return pytensor.function([theta], grad, on_unused_input="ignore")


def time_gradient(n_steps: int, n_params: int):
    t0 = time.perf_counter()
    fn = build_gradient_fn(n_steps, n_params)
    compile_s = time.perf_counter() - t0

    theta0 = np.concatenate([
        [0.45, 0.10, 0.15, 0.15, 0.15],
        [3e-4, 2e-3, 2e-3, 5e-5, 1e-3],
        [0.04, 12.0, 110.0],
        [100.0, 300.0, 200.0, 8000.0, 150.0],
    ])[:max(n_params, 13)]
    if n_params < 18:
        theta0 = theta0[:13]

    for _ in range(N_WARMUP):
        fn(theta0)

    times = []
    for _ in range(N_TIMED):
        t = time.perf_counter()
        fn(theta0)
        times.append(time.perf_counter() - t)

    return compile_s, float(np.median(times)) * 1000.0


def project(ms: float, chains=4, draws=1000, tune=1000):
    total = chains * (draws + tune)
    print(f"    {'treedepth':>10} {'grads/draw':>11} {'wall clock':>14}")
    for td in (5, 6, 7, 8):
        n = 2 ** td - 1
        hrs = total * n * ms / 1000.0 / 3600.0
        s = f"{hrs:.1f} h" if hrs < 48 else f"{hrs/24:.1f} days"
        print(f"    {td:>10} {n:>11} {s:>14}")


if __name__ == "__main__":
    print(f"machine : {platform.processor() or platform.machine()}")
    print(f"pytensor: {pytensor.__version__}")
    print(f"timing  : median of {N_TIMED} evaluations after {N_WARMUP} warmup\n")

    results = {}
    for n_params in PARAM_COUNTS:
        for n_steps in STEP_COUNTS:
            compile_s, ms = time_gradient(n_steps, n_params)
            results[(n_params, n_steps)] = ms
            print(f"{n_params} params, {n_steps} steps  ->  "
                  f"{ms:8.2f} ms/gradient   (compile {compile_s:.1f} s)")

    print("\nScaling")
    for n_params in PARAM_COUNTS:
        a = results[(n_params, STEP_COUNTS[0])]
        b = results[(n_params, STEP_COUNTS[1])]
        print(f"  {n_params} params: {STEP_COUNTS[1]}/{STEP_COUNTS[0]} steps "
              f"= {b/a:.2f}x  (linear would be "
              f"{STEP_COUNTS[1]/STEP_COUNTS[0]:.2f}x)")

    print("\nProjected sampling time, 4 chains x (1000 tune + 1000 draws)")
    for key in [(18, 3287), (13, 3287), (13, 1460)]:
        if key in results:
            print(f"\n  {key[0]} params, {key[1]} steps "
                  f"({results[key]:.2f} ms/gradient):")
            project(results[key])

    print("\nRead the treedepth=6 and 7 rows. A correlated posterior "
          "(Chuter proves ours is) sits there, not at 5.")
