#!/usr/bin/env python3
"""
generate_bodies.py
Generate N test particles for the N-body simulation.

Output format: a numpy array of shape (N, 7) with columns:
  [x, y, z, vx, vy, vz, m]

Saved to bodies.npy.
"""
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16384,
                        help="number of particles to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--out", type=str, default="bodies.npy",
                        help="output file path")
    parser.add_argument("--m-min", type=float, default=0.5)
    parser.add_argument("--m-max", type=float, default=1.5)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    bodies = np.zeros((args.n, 7), dtype=np.float32)

    # Position: uniform in [0, 1]^3
    bodies[:, 0:3] = rng.uniform(0.0, 1.0, size=(args.n, 3)).astype(np.float32)

    # Velocity: initially zero (single-step does not need velocity)
    bodies[:, 3:6] = 0.0

    # Mass: uniform in [m_min, m_max]
    bodies[:, 6] = rng.uniform(args.m_min, args.m_max, size=args.n).astype(np.float32)

    np.save(args.out, bodies)

    print(f"Generated {args.n} particles in {args.out}")
    print(f"  position range: [{bodies[:, 0:3].min():.4f}, {bodies[:, 0:3].max():.4f}]")
    print(f"  mass range:     [{bodies[:, 6].min():.4f}, {bodies[:, 6].max():.4f}]")
    print(f"  total mass:     {bodies[:, 6].sum():.2f}")

if __name__ == "__main__":
    main()
