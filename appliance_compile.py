#!/usr/bin/env python3
"""
appliance_compile.py — Compile the multistep N-body WSE-3 kernel for the
Cerebras Wafer-Scale Cluster (appliance mode) on ALCF.

This is the modern equivalent of commands.sh. Instead of invoking cslc
directly via shell, it uses cerebras.sdk.client.SdkCompiler which submits
the compile job to the Kubernetes-managed appliance, then organizes all
compile artifacts into ./compiled/<LABEL>/.

Usage:
    python3 appliance_compile.py SIDE N_LOCAL N_STEPS DT_NUM DT_DEN --label LABEL

Example (config #1, SIDE64_NL1: SIDE=64, N_LOCAL=1, N_STEPS=10, dt=0.0001):
    python3 appliance_compile.py 64 1 10 1 10000 --label SIDE64_NL1

Output (created/overwritten in ./compiled/<LABEL>/):
    artifact_path.json   — path to compile artifact on the cluster
    cs_<hash>.tar.gz     — the compile artifact (tarball)
    out/                 — extracted compile output dir
    compile.log          — stdout of this script
    compile_meta.json    — config summary
"""
import argparse
import datetime
import io
import json
import logging
import os
import shutil
import socket
import sys

from cerebras.sdk.client import SdkCompiler


WSE3_FABRIC_DIMS = os.environ.get("CSL_FABRIC_DIMS", "762,1172")
WSE3_FABRIC_OFFSETS = "4,1"


class Tee:
    """Write stdout to both terminal and an in-memory string buffer."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Compile WSE-3 N-body kernel for appliance hardware execution."
    )
    parser.add_argument("side",    type=int, help="PE per side (fabric is side x side).")
    parser.add_argument("n_local", type=int, help="Bodies per PE.")
    parser.add_argument("n_steps", type=int, help="Leapfrog iterations.")
    parser.add_argument("dt_num",  type=int, help="Numerator of dt.")
    parser.add_argument("dt_den",  type=int, help="Denominator of dt.")
    parser.add_argument(
        "--label", required=True,
        help="Configuration label (e.g. SIDE64_NL1, S2b). Output goes to "
             "./compiled/<label>/.",
    )
    parser.add_argument(
        "--aliases", default="",
        help="Comma-separated list of equivalent labels (for collapsed configs).",
    )
    parser.add_argument(
        "--source-dir", default=".",
        help="Directory containing layout.csl and pe.csl (default: current dir).",
    )
    parser.add_argument(
        "--top-file", default="layout.csl",
        help="Top-level CSL file (default: layout.csl).",
    )
    parser.add_argument(
        "--compiled-dir", default="compiled",
        help="Root dir where ./compiled/<label>/ will be created (default: compiled).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose appliance logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Set up the output dir.
    out_dir = os.path.join(args.compiled_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)

    # Capture all our stdout into a buffer so we can write a .log file at the
    # end. We still print live to the user's terminal via Tee.
    log_buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_buf)

    total_pe  = args.side * args.side
    n_total   = total_pe * args.n_local
    dt_val    = args.dt_num / args.dt_den

    cslc_opts = (
        f"--arch=wse3 "
        f"--fabric-dims={WSE3_FABRIC_DIMS} "
        f"--fabric-offsets={WSE3_FABRIC_OFFSETS} "
        f"--params=side:{args.side} "
        f"--params=N_LOCAL:{args.n_local} "
        f"--params=N_STEPS:{args.n_steps} "
        f"--params=DT_NUM:{args.dt_num} "
        f"--params=DT_DEN:{args.dt_den} "
        f"--params=MEMCPY_H2D_ID:0 "
        f"--params=MEMCPY_D2H_ID:1 "
        f"--max-inlined-iterations 500 "
        f"-o out --memcpy --channels 1"
    )

    print("=" * 64)
    print("WSE-3 N-body APPLIANCE COMPILE")
    print("=" * 64)
    print(f"  Date         : {datetime.datetime.now().isoformat()}")
    print(f"  Host         : {socket.gethostname()}")
    print(f"  Label        : {args.label}")
    if args.aliases:
        print(f"  Aliases      : {args.aliases}")
    print(f"  Source dir   : {args.source_dir}")
    print(f"  Top file     : {args.top_file}")
    print(f"  Arch         : wse3 (hardware)")
    print(f"  Fabric       : {WSE3_FABRIC_DIMS}")
    print(f"  SIDE         : {args.side}  (total PE = {total_pe})")
    print(f"  N_LOCAL      : {args.n_local}")
    print(f"  N_TOTAL      : {n_total}")
    print(f"  N_STEPS      : {args.n_steps}")
    print(f"  DT           : {args.dt_num}/{args.dt_den} = {dt_val}")
    print(f"  cslc opts    : {cslc_opts}")
    print(f"  Out dir      : {out_dir}")
    print()
    print("[appliance_compile] Submitting compile job to appliance...")

    # Submit the compile. SdkCompiler.compile() will produce the artifact
    # (a .tar.gz path) and copy the extracted output dir to the destination
    # (args.source_dir, fourth arg). We pass args.source_dir so the local
    # `out/` dir is created there as before, then we move it into out_dir.

    # with SdkCompiler(disable_version_check=True) as compiler:

    with SdkCompiler(
        disable_version_check=True,
        resource_mem=256 << 30,   # 256 GiB invece di 64 default
        resource_cpu=48000,        # 48 cores invece di 24 default
    ) as compiler:
        artifact_path = compiler.compile(
            args.source_dir,
            args.top_file,
            cslc_opts,
            args.source_dir,
        )

    print()
    print(f"[appliance_compile] Compile complete.")
    print(f"[appliance_compile] artifact_path = {artifact_path}")

    # ---- Move all compile artifacts into out_dir ----------------------------
    # The compiler leaves:
    #   - artifact_path file (a .tar.gz) — currently this is the path Cerebras
    #     returned; the actual file may be in source_dir as
    #     'cs_<hash>.tar.gz' (relative)
    #   - ./out/ extracted directory in source_dir
    #
    # We move all of them into compiled/<label>/.

    # 1) Move the tar.gz if it's local. artifact_path can be relative.
    artifact_basename = os.path.basename(artifact_path)
    local_artifact = os.path.join(args.source_dir, artifact_basename)
    if os.path.exists(local_artifact):
        dst = os.path.join(out_dir, artifact_basename)
        if os.path.abspath(local_artifact) != os.path.abspath(dst):
            shutil.move(local_artifact, dst)
            print(f"[appliance_compile] Moved {artifact_basename} -> {dst}")
        artifact_in_label = os.path.relpath(dst)
    else:
        # Cluster-side path only; leave artifact_path as-is.
        artifact_in_label = artifact_path

    # 2) Move ./out/ from source_dir into out_dir.
    src_out = os.path.join(args.source_dir, "out")
    dst_out = os.path.join(out_dir, "out")
    if os.path.exists(src_out) and os.path.abspath(src_out) != os.path.abspath(dst_out):
        if os.path.exists(dst_out):
            shutil.rmtree(dst_out)
        shutil.move(src_out, dst_out)
        print(f"[appliance_compile] Moved out/ -> {dst_out}")

    # 3) Write artifact_path.json into out_dir. We store the cluster-side
    #    artifact path (the one returned by SdkCompiler.compile), which is
    #    what SdkLauncher/SdkRuntime accept.
    apj = os.path.join(out_dir, "artifact_path.json")
    with open(apj, "w", encoding="utf8") as f:
        json.dump({"artifact_path": artifact_path}, f)
    print(f"[appliance_compile] Wrote {apj}")

    # 4) Write compile metadata.
    meta = {
        "label": args.label,
        "aliases": args.aliases.split(",") if args.aliases else [],
        "date": datetime.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "config": {
            "side": args.side,
            "n_local": args.n_local,
            "n_steps": args.n_steps,
            "dt_num": args.dt_num,
            "dt_den": args.dt_den,
            "n_total": n_total,
            "total_pe": total_pe,
        },
        "cslc_options": cslc_opts,
        "artifact_path": artifact_path,
    }
    meta_path = os.path.join(out_dir, "compile_meta.json")
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    print(f"[appliance_compile] Wrote {meta_path}")

    # Restore stdout, then dump the captured stdout to compile.log.
    sys.stdout = original_stdout
    log_path = os.path.join(out_dir, "compile.log")
    with open(log_path, "w", encoding="utf8") as f:
        f.write(log_buf.getvalue())
    print(f"[appliance_compile] Wrote {log_path}")


if __name__ == "__main__":
    main()
