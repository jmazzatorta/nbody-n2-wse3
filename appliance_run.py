#!/usr/bin/env python3
"""
appliance_run.py — Production runner for the N-body benchmark on CS-3.

This script replaces the previous SdkLauncher-based appliance_run.py (which
failed due to a cluster-server bug — CRD task missing in cluster_details for
execute jobs) with the direct SdkRuntime pattern. It also replaces the
ad-hoc run_new.py that we used for the smoke test.

The flow:
  1. Resolve the artifact (.tar.gz) from compiled/<LABEL>/.
  2. Extract it in extracted/<LABEL>/ if not already done (idempotent).
  3. Locate out/out.json inside the extracted dir to read compile-time
     parameters (side, N_LOCAL, N_STEPS, dt).
  4. Auto-resolve the bodies file from N_TOTAL if not provided.
  5. Pick the next available rep number in runs/<LABEL>_run<NN>/.
  6. Open SdkRuntime as a context manager (real CS-3 hardware by default,
     or simulator with --simulator).
  7. Push data, launch compute_step, pull results, decode timings.
  8. Save device_final.npy and results.json into runs/<LABEL>_run<NN>/.
  9. Write run.log (stdout tee) and run_meta.json (structured metadata).

Usage:
    source ~/cs_appliance_sdk/bin/activate
    cd ~/wse-n2

    # Standard use: --label resolves everything else automatically
    python appliance_run.py --label S2b

    # Explicit overrides
    python appliance_run.py --label S2b --rep 3
    python appliance_run.py --label S2b --bodies bodies/N65536/bodies.npy
    python appliance_run.py --label S2b --simulator

This script must be run with the appliance venv python (NOT cs_python).
"""
import argparse
import datetime
import glob
import io
import json
import os
import re
import socket
import sys
import tarfile
import time

import numpy as np

# NEW SDK imports (appliance venv only).
from cerebras.sdk.client import SdkRuntime
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder


# ===========================================================================
# Tee for stdout: write to terminal AND to an in-memory buffer (so we can
# dump the buffer to a log file at the end).
# ===========================================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


# ===========================================================================
# Ring topology (must match layout.csl serpentine convention)
# ===========================================================================

def ring_pos_to_pe(rp, side):
    pe_y = rp // side
    rem  = rp % side
    pe_x = rem if (pe_y % 2 == 0) else (side - 1 - rem)
    return pe_x, pe_y


def build_per_pe_tensors(bodies, side, n_local):
    """Distribute bodies in ring order. Returns local_particles, n_pred_fwd,
    n_pred_bwd, and the mapping global body index -> (pe_x, pe_y, slot)."""
    total_pe = side * side
    n_total  = bodies.shape[0]
    assert n_total == total_pe * n_local, (
        f"N_TOTAL ({n_total}) != side^2 * N_LOCAL ({total_pe} * {n_local})"
    )

    local_particles = np.zeros((side, side, n_local * 7), dtype=np.float32)
    n_pred_fwd      = np.zeros((side, side, 1),           dtype=np.uint32)
    n_pred_bwd      = np.zeros((side, side, 1),           dtype=np.uint32)

    body_to_pe = []
    L = total_pe
    for rp in range(L):
        pe_x, pe_y = ring_pos_to_pe(rp, side)
        chunk = bodies[rp * n_local : (rp + 1) * n_local]
        local_particles[pe_y, pe_x, :] = chunk.flatten()
        n_pred_fwd[pe_y, pe_x, 0] = rp
        n_pred_bwd[pe_y, pe_x, 0] = (L - 1) - rp
        for slot in range(n_local):
            body_to_pe.append((pe_x, pe_y, slot))

    return local_particles, n_pred_fwd, n_pred_bwd, body_to_pe


# ===========================================================================
# Memcpy helpers
# ===========================================================================

def h2d_per_pe_f32(runner, sym, data, side, words_per_pe):
    runner.memcpy_h2d(sym, data.reshape(-1),
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False)


def h2d_per_pe_u32(runner, sym, data, side, words_per_pe):
    runner.memcpy_h2d(sym, data.reshape(-1),
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False)


def d2h_per_pe_f32(runner, sym, side, words_per_pe):
    out = np.zeros([side * side * words_per_pe], dtype=np.float32)
    runner.memcpy_d2h(out, sym,
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False)
    return out.reshape(side, side, words_per_pe)


def d2h_per_pe_u16(runner, sym, side, words_per_pe):
    out_u32 = np.zeros([side * side * words_per_pe], dtype=np.uint32)
    runner.memcpy_d2h(out_u32, sym,
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_16BIT,
                      nonblock=False)
    out_u16 = (out_u32 & 0xFFFF).astype(np.uint16)
    return out_u16.reshape(side, side, words_per_pe)


def decode_ts48(ts_u16_arr):
    """Decode 3 little-endian u16 into a 48-bit int."""
    lo  = int(ts_u16_arr[0])
    mid = int(ts_u16_arr[1])
    hi  = int(ts_u16_arr[2])
    return lo | (mid << 16) | (hi << 32)


# ===========================================================================
# Label resolution: tarball, extracted dir, out.json, bodies
# ===========================================================================

def resolve_artifact_path(compiled_dir, label):
    """Find the single cs_*.tar.gz inside compiled/<LABEL>/."""
    pattern = os.path.join(compiled_dir, label, "cs_*.tar.gz")
    candidates = glob.glob(pattern)
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No artifact found matching {pattern}. "
            f"Did you run appliance_compile.py --label {label} first?"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple artifacts in compiled/{label}/: {candidates}. "
            f"Please clean up: only one tarball expected."
        )
    return candidates[0]


def ensure_extracted(artifact_path, extracted_dir, label):
    """Extract artifact_path into extracted/<LABEL>/ if not already done.
    Returns the path to the inner 'out' dir.
    """
    label_dir = os.path.join(extracted_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Look for an existing out dir.
    inner_out = glob.glob(os.path.join(label_dir, "cs_*", "out"))
    if inner_out:
        return inner_out[0]

    # Otherwise extract.
    print(f"[setup] Extracting {artifact_path} -> {label_dir}/")
    with tarfile.open(artifact_path, "r:gz") as tar:
        tar.extractall(path=label_dir)

    inner_out = glob.glob(os.path.join(label_dir, "cs_*", "out"))
    if not inner_out:
        raise RuntimeError(
            f"Extraction succeeded but no cs_*/out dir found in {label_dir}"
        )
    return inner_out[0]


def find_next_rep(runs_dir, label):
    """Find the lowest unused rep number for runs/<label>_run<NN>/."""
    if not os.path.isdir(runs_dir):
        return 1
    pattern = re.compile(rf"^{re.escape(label)}_run(\d+)$")
    used = []
    for name in os.listdir(runs_dir):
        m = pattern.match(name)
        if m:
            used.append(int(m.group(1)))
    return max(used) + 1 if used else 1


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the N-body benchmark on CS-3 hardware (or simulator)."
    )
    parser.add_argument("--label", required=True,
                        help="Configuration label (e.g. S2b, L64_1). "
                             "Resolves compiled/<LABEL>/cs_*.tar.gz and "
                             "extracted/<LABEL>/.../out/.")
    parser.add_argument("--bodies", default=None,
                        help="Input particles file. If omitted, auto-resolved "
                             "from N_TOTAL as bodies/N<N_TOTAL>/bodies.npy.")
    parser.add_argument("--rep", type=int, default=None,
                        help="Run repetition number (default: auto-detect next).")
    parser.add_argument("--simulator", action="store_true",
                        help="Use simulator instead of real CS-3 hardware.")
    parser.add_argument("--compiled-dir", default="compiled",
                        help="Root directory for compiled artifacts.")
    parser.add_argument("--extracted-dir", default="extracted",
                        help="Root directory for extracted tarballs.")
    parser.add_argument("--runs-dir", default="runs",
                        help="Root directory for run outputs.")
    args = parser.parse_args()

    # ---- Resolve everything from --label -----------------------------------
    artifact_path = resolve_artifact_path(args.compiled_dir, args.label)
    out_dir_extracted = ensure_extracted(
        artifact_path, args.extracted_dir, args.label
    )

    with open(os.path.join(out_dir_extracted, "out.json")) as f:
        compile_data = json.load(f)
    side     = int(compile_data["params"]["side"])
    n_local  = int(compile_data["params"]["N_LOCAL"])
    n_steps  = int(compile_data["params"]["N_STEPS"])
    dt_num   = int(compile_data["params"]["DT_NUM"])
    dt_den   = int(compile_data["params"]["DT_DEN"])
    dt       = dt_num / dt_den
    total_pe = side * side
    n_total  = total_pe * n_local

    # Auto-resolve bodies path if not given.
    if args.bodies is None:
        args.bodies = f"bodies/N{n_total}/bodies.npy"
    if not os.path.exists(args.bodies):
        sys.exit(
            f"ERROR: bodies file not found: {args.bodies}\n"
            f"Generate it first: cs_python generate_bodies.py --n {n_total} "
            f"--seed 42 --out {args.bodies}"
        )

    # Compile metadata (optional, for header info).
    compile_meta_path = os.path.join(
        args.compiled_dir, args.label, "compile_meta.json"
    )
    compile_meta = {}
    if os.path.exists(compile_meta_path):
        with open(compile_meta_path) as f:
            compile_meta = json.load(f)

    # ---- Allocate run directory --------------------------------------------
    if args.rep is None:
        rep = find_next_rep(args.runs_dir, args.label)
    else:
        rep = args.rep
    rep_name = f"{args.label}_run{rep:02d}"
    run_dir = os.path.join(args.runs_dir, rep_name)
    os.makedirs(run_dir, exist_ok=True)

    out_device_final = os.path.join(run_dir, "device_final.npy")
    out_results_json = os.path.join(run_dir, "results.json")
    out_run_log      = os.path.join(run_dir, "run.log")
    out_run_meta     = os.path.join(run_dir, "run_meta.json")

    # ---- Tee stdout so we can write run.log at the end ---------------------
    log_buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_buf)

    # ---- Header (Slurm-style) ----------------------------------------------
    start_time = datetime.datetime.now()
    print("=" * 64)
    print("=== Job info ===")
    print(f"Date: {start_time.strftime('%a %b %d %H:%M:%S %Z %Y').strip()}")
    print(f"Node: {socket.gethostname()}")
    print(f"Label: {args.label}  Rep: {rep:02d}")
    print(f"Target: {'simulator' if args.simulator else 'CS-3 hardware'}")
    print(f"Run dir: {run_dir}")
    print("=== Configuration ===")
    print(f"  Fabric         : {side} x {side} = {total_pe} PE")
    print(f"  N_LOCAL        : {n_local}  (bodies per PE)")
    print(f"  N_TOTAL        : {n_total}")
    print(f"  N_STEPS        : {n_steps}")
    print(f"  DT             : {dt_num}/{dt_den} = {dt}")
    print(f"  Artifact       : {artifact_path}")
    print(f"  Bodies         : {args.bodies}")
    print("=" * 64)
    print()

    # ---- Load bodies --------------------------------------------------------
    bodies = np.load(args.bodies)
    assert bodies.shape == (n_total, 7), (
        f"bodies shape {bodies.shape} != ({n_total}, 7). "
        f"Regenerate with --n {n_total}."
    )

    print(f"[1/4] Building per-PE tensors...")
    local_particles, n_pred_fwd, n_pred_bwd, body_to_pe = \
        build_per_pe_tensors(bodies, side, n_local)

    # Timing buffer is 9 u16 per step * (N_STEPS+1) iterations.
    time_buf_words = 9 * (n_steps + 1)

    # Pre-declare result containers (filled inside the with-block).
    particles_3d = None
    forces_3d    = None
    timing_3d    = None
    t_push       = 0.0
    t_launch     = 0.0
    t_pull       = 0.0
    error_message = None

    # ---- Open SdkRuntime ---------------------------------------------------
    print(f"[2/4] Opening SdkRuntime...")

    try:
        with SdkRuntime(
            artifact_path,
            simulator=args.simulator,
            disable_version_check=True,
        ) as runner:

            sym = {
                "local_particles":  runner.get_id("ptr_local_particles"),
                "local_forces":     runner.get_id("ptr_local_forces"),
                "done":             runner.get_id("ptr_done"),
                "n_pred_fwd":       runner.get_id("ptr_n_pred_fwd"),
                "n_pred_bwd":       runner.get_id("ptr_n_pred_bwd"),
                "time_buf_u16":     runner.get_id("ptr_time_buf_u16"),
            }

            print(f"[3/4] Initializing PE routing (runtime_init)...")
            t_init0 = time.time()
            runner.launch("runtime_init", nonblock=False)
            t_init = time.time() - t_init0

            print(f"[3/4] Pushing data...")
            t_push0 = time.time()
            h2d_per_pe_f32(runner, sym["local_particles"], local_particles, side, n_local * 7)
            h2d_per_pe_u32(runner, sym["n_pred_fwd"], n_pred_fwd, side, 1)
            h2d_per_pe_u32(runner, sym["n_pred_bwd"], n_pred_bwd, side, 1)
            t_push = time.time() - t_push0

            print(f"[4/4] Launching compute_step ({n_steps} leapfrog steps on-device)...")
            t_launch0 = time.time()
            runner.launch("compute_step", nonblock=False)
            t_launch = time.time() - t_launch0

            t_pull0 = time.time()
            particles_3d = d2h_per_pe_f32(runner, sym["local_particles"], side, n_local * 7)
            forces_3d    = d2h_per_pe_f32(runner, sym["local_forces"],    side, n_local * 3)
            timing_3d    = d2h_per_pe_u16(runner, sym["time_buf_u16"],    side, time_buf_words)
            t_pull = time.time() - t_pull0

    except Exception as e:
        error_message = repr(e)
        print(f"\n[appliance_run] ERROR: {error_message}")

    # ---- If we got results, post-process them ------------------------------
    has_results = particles_3d is not None and error_message is None

    if has_results:
        global_particles = np.zeros((n_total, 7), dtype=np.float32)
        global_forces    = np.zeros((n_total, 3), dtype=np.float32)
        for g, (pe_x, pe_y, slot) in enumerate(body_to_pe):
            global_particles[g] = particles_3d[pe_y, pe_x, slot*7:slot*7+7]
            global_forces[g]    = forces_3d[pe_y, pe_x, slot*3:slot*3+3]

        # Decode timing for every PE, every step.
        t_self_cycles     = np.zeros((total_pe, n_steps + 1), dtype=np.int64)
        t_pipeline_cycles = np.zeros((total_pe, n_steps + 1), dtype=np.int64)
        t_total_cycles    = np.zeros((total_pe, n_steps + 1), dtype=np.int64)

        for pe_y in range(side):
            for pe_x in range(side):
                buf = timing_3d[pe_y, pe_x]
                idx_pe = pe_y * side + pe_x
                for s in range(n_steps + 1):
                    base = 9 * s
                    start = decode_ts48(buf[base + 0 : base + 3])
                    self_ = decode_ts48(buf[base + 3 : base + 6])
                    end   = decode_ts48(buf[base + 6 : base + 9])
                    t_self_cycles[idx_pe, s]     = self_ - start
                    t_pipeline_cycles[idx_pe, s] = end - self_
                    t_total_cycles[idx_pe, s]    = end - start

        CLK_HZ = 850e6
        def cyc_to_ms(c): return c / CLK_HZ * 1e3
        def stats(arr):
            return int(arr.min()), int(arr.max()), int(arr.mean())

        # Per-step printout.
        print()
        print("--- DEVICE TIMING per step (cycles) ---")
        print(f"  step | t_self                          | t_pipeline                      | t_total")
        print(f"       | min        max        avg       | min        max        avg       | min        max        avg")
        print(f"  -----+---------------------------------+---------------------------------+---------------------------------")
        for s in range(n_steps):
            ts_min, ts_max, ts_avg = stats(t_self_cycles[:, s])
            tp_min, tp_max, tp_avg = stats(t_pipeline_cycles[:, s])
            tt_min, tt_max, tt_avg = stats(t_total_cycles[:, s])
            print(f"   {s:3d} | {ts_min:9d} {ts_max:9d} {ts_avg:9d} | "
                  f"{tp_min:9d} {tp_max:9d} {tp_avg:9d} | "
                  f"{tt_min:9d} {tt_max:9d} {tt_avg:9d}")

        self_all = t_self_cycles[:, :n_steps].flatten()
        pipe_all = t_pipeline_cycles[:, :n_steps].flatten()
        tot_all  = t_total_cycles[:, :n_steps].flatten()
        self_min, self_max, self_avg = stats(self_all)
        pipe_min, pipe_max, pipe_avg = stats(pipe_all)
        tot_min,  tot_max,  tot_avg  = stats(tot_all)

        print()
        print("--- AGGREGATE (steps 0..N_STEPS-1) ---")
        print(f"  t_self     : min={self_min:>10d}  max={self_max:>10d}  avg={self_avg:>10d}   ({cyc_to_ms(self_avg):.3f} ms avg)")
        print(f"  t_pipeline : min={pipe_min:>10d}  max={pipe_max:>10d}  avg={pipe_avg:>10d}   ({cyc_to_ms(pipe_avg):.3f} ms avg)")
        print(f"  t_total    : min={tot_min:>10d}   max={tot_max:>10d}   avg={tot_avg:>10d}    ({cyc_to_ms(tot_avg):.3f} ms avg)")

        print()
        print(f"--- WALL-CLOCK (host side) ---")
        print(f"  Push data       : {t_push:.3f} s")
        print(f"  Launch          : {t_launch:.3f} s")
        print(f"  Pull results    : {t_pull:.3f} s")

        n_pairs = n_total * (n_total - 1)
        pairs_per_sec = (n_pairs / (tot_max / CLK_HZ)) if tot_max > 0 else None
        if pairs_per_sec:
            print()
            print(f"--- THROUGHPUT (1 step, device-only, from max t_total) ---")
            print(f"  Pair interactions per step : {n_pairs:.3e}")
            print(f"  Pairs/sec                  : {pairs_per_sec:.3e}")

        n_nonzero = int(np.any(global_forces != 0.0, axis=1).sum())
        print()
        print(f"--- SANITY ---")
        print(f"  Bodies with non-zero force : {n_nonzero}/{n_total}")

        # Save device final state (compatible with reference.npy from cuda-ref).
        device_state = np.concatenate([
            global_particles[:, 0:3],
            global_particles[:, 3:6],
            global_particles[:, 6:7],
            global_forces[:, 0:3],
        ], axis=1).astype(np.float32)
        np.save(out_device_final, device_state)
        print(f"  Saved {out_device_final} (shape {device_state.shape})")

        # Structured results JSON.
        per_step = []
        for s in range(n_steps):
            ts_min, ts_max, ts_avg = stats(t_self_cycles[:, s])
            tp_min, tp_max, tp_avg = stats(t_pipeline_cycles[:, s])
            tt_min, tt_max, tt_avg = stats(t_total_cycles[:, s])
            per_step.append({
                "step": s,
                "t_self":     {"min": ts_min, "max": ts_max, "avg": ts_avg},
                "t_pipeline": {"min": tp_min, "max": tp_max, "avg": tp_avg},
                "t_total":    {"min": tt_min, "max": tt_max, "avg": tt_avg},
            })

        results = {
            "config": {
                "side": side,
                "n_local": n_local,
                "n_total": n_total,
                "total_pe": total_pe,
                "n_steps": n_steps,
                "dt_num": dt_num,
                "dt_den": dt_den,
                "dt": dt,
            },
            "per_step_cycles": per_step,
            "aggregate_cycles": {
                "t_self":     {"min": self_min, "max": self_max, "avg": self_avg},
                "t_pipeline": {"min": pipe_min, "max": pipe_max, "avg": pipe_avg},
                "t_total":    {"min": tot_min,  "max": tot_max,  "avg": tot_avg},
            },
            "ms_at_850MHz": {
                "t_self_avg":     cyc_to_ms(self_avg),
                "t_pipeline_avg": cyc_to_ms(pipe_avg),
                "t_total_avg":    cyc_to_ms(tot_avg),
            },
            "host_wallclock_s": {
                "push":   t_push,
                "launch": t_launch,
                "pull":   t_pull,
            },
            "throughput": {
                "n_pairs_per_step": n_pairs,
                "pairs_per_sec_est": pairs_per_sec,
            },
            "sanity": {
                "n_bodies_nonzero_force": n_nonzero,
            },
        }
        with open(out_results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {out_results_json}")

    # ---- Footer (Slurm-style) ----------------------------------------------
    end_time = datetime.datetime.now()
    print()
    print("=== Done ===")
    print(f"End: {end_time.strftime('%a %b %d %H:%M:%S %Z %Y').strip()}")
    print(f"Wall-clock duration: {(end_time - start_time).total_seconds():.2f} s")
    print(f"Status: {'FAILED (' + error_message + ')' if error_message else 'OK'}")
    print("=" * 64)

    # Restore stdout and dump log.
    sys.stdout = original_stdout
    with open(out_run_log, "w") as f:
        f.write(log_buf.getvalue())
    print(f"[appliance_run] Wrote {out_run_log}")

    # Run metadata.
    meta = {
        "label": args.label,
        "rep": rep,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_s": (end_time - start_time).total_seconds(),
        "host": socket.gethostname(),
        "target": "simulator" if args.simulator else "CS-3 hardware",
        "artifact_path": artifact_path,
        "bodies_file": args.bodies,
        "compile_meta": compile_meta,
        "status": "FAILED" if error_message else "OK",
        "error": error_message,
        "files": {
            "device_final": out_device_final if os.path.exists(out_device_final) else None,
            "results_json": out_results_json if os.path.exists(out_results_json) else None,
            "run_log":      out_run_log,
        },
    }
    with open(out_run_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[appliance_run] Wrote {out_run_meta}")

    if error_message:
        sys.exit(1)


if __name__ == "__main__":
    main()
