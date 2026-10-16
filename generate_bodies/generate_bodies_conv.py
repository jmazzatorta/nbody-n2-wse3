#!/usr/bin/env python3
"""
generate_bodies_conv.py
Same as generate_bodies.py, but with initial velocities drawn from a Gaussian
distribution, with net momentum removed. Used for the O(DT^2) convergence
test where a non-degenerate initial state is required.
"""
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=43,
                        help="random seed (default 43, different from production seed 42)")
    parser.add_argument("--out", type=str, default="bodies.npy")
    parser.add_argument("--m-min", type=float, default=0.5)
    parser.add_argument("--m-max", type=float, default=1.5)
    parser.add_argument("--v-sigma", type=float, default=0.1,
                        help="stddev of initial velocity distribution")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bodies = np.zeros((args.n, 7), dtype=np.float32)

    # Position: uniform in [0, 1]^3
    bodies[:, 0:3] = rng.uniform(0.0, 1.0, size=(args.n, 3)).astype(np.float32)

    # Mass: uniform in [m_min, m_max]
    bodies[:, 6] = rng.uniform(args.m_min, args.m_max, size=args.n).astype(np.float32)

    # Velocity: Gaussian with sigma=v_sigma, then subtract center-of-mass momentum
    v = rng.normal(0.0, args.v_sigma, size=(args.n, 3)).astype(np.float32)
    com_v = (bodies[:, 6:7] * v).sum(axis=0) / bodies[:, 6].sum()
    v = v - com_v.astype(np.float32)
    bodies[:, 3:6] = v

    np.save(args.out, bodies)
    print(f"Generated {args.n} particles in {args.out}")
    print(f"  position range: [{bodies[:, 0:3].min():.4f}, {bodies[:, 0:3].max():.4f}]")
    print(f"  velocity range: [{bodies[:, 3:6].min():.4f}, {bodies[:, 3:6].max():.4f}]")
    print(f"  |v| max:        {np.abs(bodies[:, 3:6]).max():.4f}")
    print(f"  mass range:     [{bodies[:, 6].min():.4f}, {bodies[:, 6].max():.4f}]")
    print(f"  total mass:     {bodies[:, 6].sum():.2f}")

    # Sanity: total momentum should be ~0
    P = (bodies[:, 6:7] * bodies[:, 3:6]).sum(axis=0)
    print(f"  net momentum:   {P}  (|P|={np.linalg.norm(P):.2e})")
    K = 0.5 * np.sum(bodies[:, 6] * np.sum(bodies[:, 3:6]**2, axis=1))
    print(f"  kinetic energy: {K:.4e}")

if __name__ == "__main__":
    main()
