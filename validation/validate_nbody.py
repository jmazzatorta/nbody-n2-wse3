
#!/usr/bin/env python3
"""
validate_nbody.py — Correctness validation for the WSE-3 N-body kernel that does
NOT rely on trajectory-by-trajectory equality with a CPU reference.

Rationale
---------
A gravitational N-body system is chaotic (Lyapunov instability): nearby states
diverge exponentially, so any fp32 rounding difference vs a CPU reference is
amplified exponentially over many steps. Comparing final positions after many
steps is therefore meaningless as a correctness test. We validate instead the
physical invariants and the order of the integrator, which are invariant to
rounding order.

METHOD 1 — Conserved invariants (needs only initial + final state)
    * Total energy  E = K + U   (leapfrog is symplectic: bounded drift, no growth).
      U uses the SAME softening eps^2 as the kernel.
    * Total linear momentum  P   (Newton's 3rd law).
    * Total angular momentum  L  (central forces).

METHOD 2 — Order of convergence (needs several finals at decreasing DT, same T,
           same initial bodies). PRIMARY estimator is REFERENCE-FREE
           self-convergence: for a p-th order method,
                ||x(dt) - x(dt/2)|| ~ dt^p,
           so the slope of log||Δx|| vs log(dt) is p (~2 for leapfrog). This
           avoids both failure modes of a fixed-reference fit: coarse points
           outside the asymptotic regime, and contamination of the point nearest
           the reference. A progressive fit (dropping the coarsest points) shows
           the observed order settling onto its asymptotic value.

A third check — one-step force accuracy vs CPU (mean rel err ~2.4e-6) — is a
separate workflow that validates the pointwise force computation; not repeated here.

Usage
-----
  # Invariants only (one run):
  python validate_nbody.py --bodies bodies/N4096/bodies.npy \
                           --final runs/RUN/device_final.npy

  # Invariants + convergence (several runs at decreasing DT):
  python validate_nbody.py --bodies bodies/N4096_conv/bodies.npy \
                           --final runs/CONV_016/device_final.npy \
                                   runs/CONV_032/device_final.npy \
                                   runs/CONV_064/device_final.npy \
                                   runs/CONV_128/device_final.npy \
                                   runs/CONV_256/device_final.npy

  DT / N_STEPS are auto-read from the sibling results.json when present.
"""

import argparse
import json
import os
import sys

import numpy as np


# --- Physical constants: MUST match the kernel (constants.csl). ---
DEFAULT_G      = 1.0       # G_GRAV
DEFAULT_EPS_SQ = 0.0001    # SOFT_EPS_SQ (Plummer softening, eps^2)

BODIES_NCOL = 7    # [x,y,z, vx,vy,vz, m]
FINAL_NCOL  = 10   # [x,y,z, vx,vy,vz, m, fx,fy,fz]


# ===========================================================================
# Loading
# ===========================================================================

def load_state(path):
    arr = np.load(path)
    if arr.ndim != 2:
        sys.exit(f"ERROR: {path} has ndim {arr.ndim}, expected 2D.")
    ncol = arr.shape[1]
    if ncol not in (BODIES_NCOL, FINAL_NCOL):
        sys.exit(f"ERROR: {path} has {ncol} cols; expected {BODIES_NCOL} or "
                 f"{FINAL_NCOL}. Adjust column constants at top of script if needed.")
    a = arr.astype(np.float64)
    return a[:, 0:3], a[:, 3:6], a[:, 6]     # pos, vel, mass


def read_run_meta(final_path):
    rj = os.path.join(os.path.dirname(final_path), "results.json")
    if os.path.exists(rj):
        try:
            with open(rj) as f:
                cfg = json.load(f).get("config", {})
            return cfg.get("dt"), cfg.get("n_steps")
        except Exception:
            pass
    return None, None


# ===========================================================================
# Invariants (float64)
# ===========================================================================

def kinetic_energy(vel, mass):
    return 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))


def potential_energy(pos, mass, G, eps_sq, block=256):
    """U = -G sum_{i<j} m_i m_j / sqrt(|r_i-r_j|^2 + eps^2), row-blocked, direct
    coordinate subtraction (stable for close pairs)."""
    N = pos.shape[0]
    total = 0.0
    for s in range(0, N, block):
        e = min(s + block, N)
        d = pos[s:e, None, :] - pos[None, :, :]
        r2 = np.sum(d * d, axis=2) + eps_sq
        contrib = (mass[s:e, None] * mass[None, :]) / np.sqrt(r2)
        contrib[np.arange(e - s), np.arange(s, e)] = 0.0   # drop self
        total += float(np.sum(contrib))
    return -G * 0.5 * total


def total_momentum(vel, mass):
    return np.sum(mass[:, None] * vel, axis=0)


def total_angular_momentum(pos, vel, mass):
    return np.sum(mass[:, None] * np.cross(pos, vel), axis=0)


def momentum_scale(vel, mass):
    return float(np.sum(mass * np.linalg.norm(vel, axis=1)))


def angmom_scale(pos, vel, mass):
    return float(np.sum(mass * np.linalg.norm(np.cross(pos, vel), axis=1)))


def invariants_report(bodies_path, final_path, G, eps_sq):
    pos0, vel0, m0 = load_state(bodies_path)
    posT, velT, mT = load_state(final_path)
    if pos0.shape[0] != posT.shape[0]:
        sys.exit(f"ERROR: N mismatch {pos0.shape[0]} vs {posT.shape[0]}.")

    K0, KT = kinetic_energy(vel0, m0), kinetic_energy(velT, mT)
    print("  computing potential energy (O(N^2))...", flush=True)
    U0, UT = potential_energy(pos0, m0, G, eps_sq), potential_energy(posT, mT, G, eps_sq)
    E0, ET = K0 + U0, KT + UT
    P0, PT = total_momentum(vel0, m0), total_momentum(velT, mT)
    L0, LT = total_angular_momentum(pos0, vel0, m0), total_angular_momentum(posT, velT, mT)

    e_rel = abs(ET - E0) / max(abs(E0), 1e-300)
    p_abs = float(np.linalg.norm(PT - P0))
    l_abs = float(np.linalg.norm(LT - L0))

    P0_mag, PT_mag = float(np.linalg.norm(P0)), float(np.linalg.norm(PT))
    L0_mag, LT_mag = float(np.linalg.norm(L0)), float(np.linalg.norm(LT))

    # --- Normalization scales ---
    # Momentum: initial conditions are centered, so |P0| ~ 0 by construction and
    # is NOT a usable denominator; the natural scale is the total momentum in the
    # system, sum_i m_i|v_i|.
    p_budget = max(momentum_scale(vel0, m0), momentum_scale(velT, mT), 1e-300)
    # Angular momentum: the system carries a real net L, so the physically correct
    # relative drift is |dL|/|L0|. We do NOT normalize by sum_i m_i|r_i x v_i|,
    # which suffers heavy internal cancellation (individual r_i x v_i are large but
    # their vector sum |L| is small) and would drastically understate the drift.
    l_budget = max(angmom_scale(pos0, vel0, m0), angmom_scale(posT, velT, mT), 1e-300)
    l_ref = max(L0_mag, LT_mag)
    if l_ref > 1e-6 * l_budget:            # L0 is a real quantity, not cancellation noise
        l_scale, l_label = l_ref, "|L0|"
    else:                                  # near-zero net L: fall back to the budget scale
        l_scale, l_label = l_budget, "sum|m r x v| (|L0|~0)"

    rest_start = momentum_scale(vel0, m0) < 1e-9 * p_budget
    if rest_start:
        print("  [warn] initial condition starts from REST (v0 ~ 0). Energy converts"
              " potential->kinetic during collapse; momentum/angular-momentum")
        print("         drift is normalized to the run's velocity scale, but for a"
              " clean invariants test prefer non-zero initial velocities.")

    print()
    print(f"  --- ENERGY ---")
    print(f"    K:  init={K0:+.8e}   final={KT:+.8e}")
    print(f"    U:  init={U0:+.8e}   final={UT:+.8e}")
    print(f"    E:  init={E0:+.8e}   final={ET:+.8e}")
    print(f"    relative energy drift |dE|/|E0| = {e_rel:.3e}")
    print(f"  --- LINEAR MOMENTUM ---")
    print(f"    |P0|={P0_mag:.3e}   |PT|={PT_mag:.3e}")
    print(f"    |dP| = {p_abs:.3e}   (relative to sum|m v| = {p_abs / p_budget:.3e})")
    print(f"  --- ANGULAR MOMENTUM ---")
    print(f"    |L0|={L0_mag:.3e}   |LT|={LT_mag:.3e}")
    print(f"    |dL| = {l_abs:.3e}   (relative to {l_label} = {l_abs / l_scale:.3e})")

    return {"energy": {"E0": E0, "ET": ET, "rel_drift": e_rel},
            "momentum": {"abs_drift": p_abs, "rel_drift_vs_budget": p_abs / p_budget},
            "angular_momentum": {"abs_drift": l_abs, "L0": L0_mag, "LT": LT_mag,
                                 "rel_drift": l_abs / l_scale, "rel_scale": l_label},
            "rest_start": bool(rest_start)}


# ===========================================================================
# Convergence — reference-free self-convergence + progressive asymptotic fit
# ===========================================================================

def convergence_report(final_paths, dts, out_png=None, tol=0.15):
    data = sorted([(float(dt), load_state(p)[0], p) for dt, p in zip(dts, final_paths)],
                  key=lambda t: t[0])                       # ascending dt (finest first)
    dts_s = [d[0] for d in data]
    pos_s = [d[1] for d in data]
    path_s = [d[2] for d in data]
    n = len(data)
    for i in range(1, n):
        if pos_s[i].shape != pos_s[0].shape:
            print(f"  [skip convergence] shape mismatch across runs.")
            return None

    print()
    print("  --- ORDER OF CONVERGENCE ---")
    print("  Primary estimator: REFERENCE-FREE self-convergence.")
    print("  For a p-th order method, ||x(dt) - x(dt/2)|| ~ dt^p  =>  slope p (~2 leapfrog).")

    # Consecutive-refinement differences: D_k = RMS(x_k - x_{k+1}), at coarser dt.
    Ddt, Dval, ratios = [], [], []
    for k in range(n - 1):
        rms = float(np.sqrt(np.mean(np.sum((pos_s[k] - pos_s[k + 1]) ** 2, axis=1))))
        Ddt.append(dts_s[k + 1])          # coarser dt of the pair
        Dval.append(rms)
        ratios.append(dts_s[k + 1] / dts_s[k])
    if any(abs(r - 2.0) > 0.05 for r in ratios):
        print(f"  [note] refinement ratios not all 2x: {[f'{r:.2f}' for r in ratios]} "
              f"(local-order formula still uses actual dt ratios).")

    # Per-interval local observed order (between consecutive difference points).
    print()
    print("  reference-free consecutive differences:")
    print(f"    {'dt (coarser)':>14} {'||dx|| RMS':>14} {'local order':>12}")
    local = []
    for i in range(len(Dval)):
        slope = (float(np.log(Dval[i] / Dval[i - 1]) / np.log(Ddt[i] / Ddt[i - 1]))
                 if i > 0 else float("nan"))
        if i > 0:
            local.append((Ddt[i], slope))
        print(f"    {Ddt[i]:14.3e} {Dval[i]:14.6e} {slope:12.3f}")

    # Headline: observed order at the finest interval (most asymptotic, clean).
    richardson = local[0][1] if local else float("nan")

    # Progressive fit: drop the coarsest difference point and refit; the order
    # settles onto its asymptotic value as coarse (non-asymptotic) points leave.
    logdt = np.log(np.asarray(Ddt))
    logD = np.log(np.asarray(Dval))
    print()
    print("  progressive fit (drop coarsest points to isolate asymptotic regime):")
    print(f"    {'#pts':>5} {'dt range (fine..coarse)':>28} {'fitted order':>14}")
    fits = []
    m = len(Ddt)
    while m >= 2:
        s = float(np.polyfit(logdt[:m], logD[:m], 1)[0])
        fits.append((m, s))
        rng = f"[{Ddt[0]:.2e} .. {Ddt[m-1]:.2e}]"
        print(f"    {m:5d} {rng:>28} {s:14.3f}")
        m -= 1

    # Recommended asymptotic order.
    #  (1) If consecutive progressive fits agree within tol, the order has
    #      stabilized -> report that (coarse, non-asymptotic points dropped).
    #  (2) Otherwise DO NOT fall back to the finest-interval 2-point slope, which
    #      is a single interval with no redundancy and is statistically noisy.
    #      Prefer the finest fit that still uses >= 3 difference points (a genuine
    #      regression). Only if fewer than 3 points exist do we use the 2-point
    #      value, with an explicit insufficient-data warning.
    # fits is ordered by decreasing #points: (all, .), (all-1, .), ..., (2, .).
    # Walk adjacent pairs from most points down; the first pair that agrees within
    # tol marks the asymptotic regime. Report the MORE-robust member (more points).
    recommended, why, thin = None, "", False
    for i in range(len(fits) - 1):
        m_hi, s_hi = fits[i]        # more points
        m_lo, s_lo = fits[i + 1]    # fewer points
        if abs(s_hi - s_lo) < tol:
            recommended = s_hi
            if m_hi >= 3:
                why = (f"stabilized (within tol={tol}); reporting the {m_hi}-point "
                       f"fit (coarser out-of-regime points dropped)")
            else:
                why = (f"only a {m_hi}-point estimate agrees; indicative, not a "
                       f"robust fit -- add finer runs")
                thin = True
            break
    if recommended is None:
        robust = [(m, s) for (m, s) in fits if m >= 3]
        if robust:
            m_r, recommended = robust[-1]                     # finest >=3-point fit
            why = (f"no stabilization within tol={tol}; reporting finest robust fit "
                   f"({m_r} points). Add finer runs to tighten toward 2.0")
            thin = True
        else:
            recommended, thin = richardson, True
            why = ("only a 2-point (single-interval) estimate available; NOT a "
                   "fitted order. Add finer runs before quoting a convergence order")

    print()
    print(f"  finest-interval self-convergence (noisy, single interval) = {richardson:.3f}")
    print(f"  RECOMMENDED asymptotic order                             = {recommended:.3f}")
    print(f"    -> {why}")
    print(f"    (expected 2.0 for a correct 2nd-order leapfrog, half-step included)")
    if thin:
        print("    [advice] asymptotic regime rests on few points; 1-2 finer runs "
              "(e.g. dt/2, dt/4 of the current finest) would make the order robust.")

    if out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.loglog(Ddt, Dval, "o-", label="self-convergence ||dx||")
            # slope-2 guide through the finest point
            c = Dval[0] / (Ddt[0] ** 2)
            guide = c * np.asarray(Ddt) ** 2
            ax.loglog(Ddt, guide, "--", color="gray", label="slope 2 (reference)")
            ax.set_xlabel("DT (coarser of pair)")
            ax.set_ylabel("RMS position difference between consecutive refinements")
            ax.set_title(f"Self-convergence (asymptotic order ~ {recommended:.2f})")
            ax.legend(); ax.grid(True, which="both", ls=":")
            fig.tight_layout(); fig.savefig(out_png, dpi=140)
            print(f"    saved plot -> {out_png}")
        except ModuleNotFoundError:
            print("    [note] matplotlib not installed; skipped plot "
                  "(pip install matplotlib, or pass --plot '').")
        except Exception as ex:
            print(f"    [warn] plot failed: {ex!r}")

    return {"consecutive_dt": Ddt, "consecutive_rms": Dval,
            "local_orders": [s for _, s in local],
            "progressive_fits": fits,
            "headline_order": richardson, "recommended_order": recommended}


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--final", required=True, nargs="+")
    ap.add_argument("--dt", type=float, nargs="+", default=None)
    ap.add_argument("--nsteps", type=int, nargs="+", default=None)
    ap.add_argument("--G", type=float, default=DEFAULT_G)
    ap.add_argument("--eps-sq", type=float, default=DEFAULT_EPS_SQ)
    ap.add_argument("--tol", type=float, default=0.25,
                    help="Stability tolerance for the asymptotic-order fit.")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--plot", default="convergence.png",
                    help="Convergence plot path; pass '' to skip.")
    args = ap.parse_args()

    for p in [args.bodies] + args.final:
        if not os.path.exists(p):
            sys.exit(f"ERROR: file not found: {p}")

    print("=" * 70)
    print("N-body validation  (invariants + order of convergence)")
    print(f"  G = {args.G},  eps^2 = {args.eps_sq}")
    print(f"  bodies (initial) : {args.bodies}")
    print("=" * 70)

    results = {"config": {"G": args.G, "eps_sq": args.eps_sq, "bodies": args.bodies},
               "runs": []}

    dts, nsteps = [], []
    for i, fp in enumerate(args.final):
        dt = args.dt[i] if args.dt and i < len(args.dt) else None
        ns = args.nsteps[i] if args.nsteps and i < len(args.nsteps) else None
        if dt is None or ns is None:
            mdt, mns = read_run_meta(fp)
            dt = dt if dt is not None else mdt
            ns = ns if ns is not None else mns
        dts.append(dt); nsteps.append(ns)

    for fp, dt, ns in zip(args.final, dts, nsteps):
        print()
        print(f"[invariants] {fp}   (dt={dt}, n_steps={ns})")
        inv = invariants_report(args.bodies, fp, args.G, args.eps_sq)
        results["runs"].append({"final": fp, "dt": dt, "n_steps": ns, "invariants": inv})

    if len(args.final) >= 2:
        if any(d is None for d in dts):
            print("\n[convergence] skipped: DT unknown for some runs "
                  "(pass --dt or ensure results.json is present).")
        else:
            if all(n is not None for n in nsteps):
                Ts = [d * n for d, n in zip(dts, nsteps)]
                if max(Ts) / min(Ts) > 1.01:
                    print(f"\n  [warn] DT*N_STEPS not constant: {Ts}. Convergence "
                          f"assumes all runs reach the SAME physical time T.")
            conv = convergence_report(args.final, dts,
                                      out_png=(args.plot or None), tol=args.tol)
            results["convergence"] = conv

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n[out] wrote {args.json_out}")

    print()
    print("=" * 70)
    print("Interpretation")
    print("  * energy rel drift falls ~4x per DT halving -> O(DT^2), integrator OK")
    print("  * |dP|, |dL| small vs their scale           -> ring/forces symmetric")
    print("  * asymptotic observed order ~ 2.0           -> leapfrog 2nd order OK")
    print("  Coarsest DT may sit outside the asymptotic regime (order < 2); the")
    print("  progressive fit isolates the finer, asymptotic points automatically.")
    print("=" * 70)


if __name__ == "__main__":
    main()
