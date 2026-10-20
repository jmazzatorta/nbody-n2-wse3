# nbody-n2-wse3

Direct **O(N²) gravitational N-body** solver on the **Cerebras WSE-3**, written
in CSL. Every pair interaction is computed exactly (no tree, no multipole
approximation); the particles are distributed one chunk per PE and circulated
around the whole fabric on a **bidirectional serpentine ring**, so that each PE
sees every other PE's chunk exactly once per step. Motion is advanced with a
**leapfrog (kick-drift-kick) integrator** for `N_STEPS` steps entirely on
device.

> **Target environment.** The compile and run scripts target the **appliance
> (Wafer-Scale Cluster) version of the Cerebras SDK** as deployed on the
> **Argonne Leadership Computing Facility (ALCF) Cerebras CS-3 cluster**. They
> drive the Python appliance client (`cerebras.sdk.client`, `SdkCompiler` /
> `SdkRuntime`) rather than the containerised (Singularity) `cslc` /
> `cs_python --cmaddr` flow. See [Environment](#environment).

---

## How it works

### Mapping and the ring

The fabric is a `side × side` square of PEs. Each PE owns `N_LOCAL` bodies, so
the total is `N_TOTAL = side² · N_LOCAL`. The PEs are threaded into a single
**Hamiltonian ring of length `side²`** by a serpentine (boustrophedon) walk:
even rows run left→right, odd rows right→left, with a vertical step at the row
turn. This keeps every ring neighbour (`k` and `k±1`) **physically adjacent** on
the mesh, so every ring hop is a single-hop transfer between neighbouring PEs —
no fabric switches, no multi-hop routes.

```
 serpentine ring (side = 4, k shown)

   0 → 1 → 2 → 3
               ↓
   7 ← 6 ← 5 ← 4
   ↓
   8 → 9 → 10 → 11
                ↓
  15 ← 14 ← 13 ← 12
```

### Two counter-rotating streams

To halve the number of ring laps, each PE's chunk travels **both ways** at once:

- a **FWD** stream that moves `k → k+1 → …` around the ring, and
- a **BWD** stream that moves `k → k-1 → …`.

A PE at ring position `k` therefore receives `k` chunks from its FWD-predecessor
side and `side²-1-k` from its BWD side; together that is every other chunk in
the system. The host loads these two expected counts into `n_pred_fwd` /
`n_pred_bwd` per PE, which is how each PE knows locally when it is done (no
global barrier).

Each stream uses **two colors** with a parity convention (`k` even vs odd) so
that a sender's out-color always equals the receiver's in-color on every hop;
4 streams-directions × 2 = the 8 colors (`FWD_A/B`, `BWD_A/B`) configured
statically in `pe.csl`. A PE that receives a chunk **re-emits it** to the next
ring position (store-and-forward) and, in parallel, folds it into its local
force accumulation.

### Per-step compute (fully vectorized)

Within a step each PE:

1. Converts its bodies from the host AoS layout to SoA, computes `inv_m`.
2. Accumulates **self-forces** (its own `N_LOCAL` bodies against each other).
3. Packs its `[x, y, z, m]` chunk and injects it into both ring streams.
4. On every arriving chunk, runs `pair_update_single_source` — a vectorized
   kernel over the `N_LOCAL` local targets using fabric DSDs and `@fmacs` /
   `@map(sqrt)` — accumulating `f = G · mᵢ mⱼ (rⱼ-rᵢ) / (r² + ε²)^{3/2}`.
5. Once both streams have delivered their expected counts, `task_integrate`
   advances velocities and positions by the leapfrog update and starts the next
   step; the first and last steps apply the half-step kick that makes the scheme
   properly second-order and time-reversible.

Physics constants live in `constants.csl`: `G = 1.0` and Plummer softening
`ε² = 1e-4` (`SOFT_EPS_SQ`), which the CPU validator mirrors exactly.

### Timing

Every PE records a 48-bit cycle timestamp (3×u16) at three points per step —
start, after self-forces, and at integrate — so the host can report per-step
`t_self`, `t_pipeline` and `t_total` (min/max/avg over all PEs), in cycles and
in ms at 850 MHz, plus an estimated pair-interactions/second throughput.

---

## Repository layout

| file | role |
|---|---|
| `layout.csl` | Top-level layout: `side × side` rectangle, per-PE serpentine ring position `k`, the FWD in/out directions, and the `has_*` edge flags, all resolved at compile time |
| `pe.csl` | Per-PE program: ring transport state machine (drain/re-emit, parity colors, finalize gate), vectorized pair force kernel, leapfrog integrator, cycle timing |
| `constants.csl` | Physics (`G`, `ε²`), buffer geometry, queue IDs, task IDs |
| `appliance_compile.py` | Submits a compile job to the appliance → `compiled/<label>/` (artifact, `out/`, metadata) |
| `appliance_run.py` | Loads bodies, distributes them in ring order, runs on CS-3 (or simulator), pulls results + timing, writes `runs/<label>_run<NN>/` |
| `generate_bodies/generate_bodies.py` | Produce an `(N, 7)` bodies file `[x,y,z,vx,vy,vz,m]` (uniform cube, zero initial velocity) |
| `generate_bodies/generate_bodies_conv.py` | Variant with Gaussian, net-momentum-removed velocities, for the convergence test |
| `validation/validate_nbody.py` | Correctness by physical **invariants** (energy, linear & angular momentum) and **order of convergence** — not trajectory equality |
| `validation/n2_cpu.py` | Bit-order-faithful fp32 CPU replica of the kernel (same ring reduction order) for one-step force checks |
| `validation/pe_validate.csl` | A pe.csl variant used in the force-accuracy validation workflow |

---

## Parameters

All are compile-time CSL `param`s (one compile per configuration), passed
positionally to `appliance_compile.py`:

| arg | CSL param | meaning |
|---|---|---|
| `side` | `side` | PEs per side; fabric is `side × side` |
| `n_local` | `N_LOCAL` | bodies per PE (`N_TOTAL = side² · N_LOCAL`) |
| `n_steps` | `N_STEPS` | leapfrog steps run on device |
| `dt_num`, `dt_den` | `DT_NUM`, `DT_DEN` | timestep as an exact rational `dt = dt_num / dt_den` |

---

## Environment

The compile and run scripts target the **appliance (Wafer-Scale Cluster) SDK**,
the mode used on the **Argonne / ALCF Cerebras CS-3 cluster**, and were adapted
to that deployment. They assume:

| item | value |
|---|---|
| client module | `cerebras.sdk.client` (`SdkCompiler`, `SdkRuntime`) |
| Python env | an appliance venv, e.g. `cs_appliance_sdk` (activated with `source ~/cs_appliance_sdk/bin/activate`); **not** `cs_python` |
| architecture | `--arch=wse3` |
| fabric dims | `--fabric-dims=762,1172 --fabric-offsets=4,1` (full CS-3 wafer; override with `CSL_FABRIC_DIMS`) |
| memcpy | `--memcpy --channels 1`, H2D color 0 / D2H color 1 |
| compile resources | `resource_mem = 256 GiB`, `resource_cpu = 48000` (raised from the 64 GiB / 24-core defaults for large `side`) |

Notes specific to the appliance / ALCF setup:

- **Direct `SdkRuntime`, no `--cmaddr`.** `appliance_run.py` opens the compiled
  artifact with `SdkRuntime(artifact, simulator=False)`; the cluster resolves
  the CS-3 allocation. This replaces an earlier `SdkLauncher`-based runner that
  hit a cluster-server bug (missing CRD task in `cluster_details` for execute
  jobs).
- **Version check disabled.** Both scripts pass `disable_version_check=True` to
  get past the appliance-client / cluster-server `InconsistentVersion` warning,
  as is standard on the ALCF system.
- **Simulator fallback.** `appliance_run.py --simulator` runs the same artifact
  through the SDK fabric simulator; the only compile difference between CS-3 and
  the simulator is `--fabric-dims`.

---

## Running

Requirements: access to the Argonne / ALCF Cerebras CS-3 appliance (or another
WSC), the `cerebras.sdk` Python client in an activated appliance venv, and
NumPy (plus Matplotlib for the optional convergence plot).

### 1. Generate a bodies file

`N_TOTAL` must equal `side² · N_LOCAL`. `appliance_run.py` auto-resolves the
path `bodies/N<N_TOTAL>/bodies.npy`, so generate it there:

```bash
# e.g. side=64, N_LOCAL=1  ->  N_TOTAL = 4096
python generate_bodies/generate_bodies.py --n 4096 --seed 42 \
       --out bodies/N4096/bodies.npy
```

For the convergence test, use the non-degenerate (non-zero velocity) generator:

```bash
python generate_bodies/generate_bodies_conv.py --n 4096 --seed 43 \
       --out bodies/N4096_conv/bodies.npy
```

### 2. Compile

```bash
# positional: side n_local n_steps dt_num dt_den
python appliance_compile.py 64 1 10 1 10000 --label SIDE64_NL1
```

Output goes to `compiled/<label>/` (`cs_<hash>.tar.gz`, `out/`,
`artifact_path.json`, `compile_meta.json`, `compile.log`). For long compiles
over SSH, run inside `tmux`.

### 3. Run

```bash
source ~/cs_appliance_sdk/bin/activate
python appliance_run.py --label SIDE64_NL1            # real CS-3
python appliance_run.py --label SIDE64_NL1 --simulator
python appliance_run.py --label SIDE64_NL1 --rep 3    # explicit repetition
```

Each run resolves the artifact, extracts it under `extracted/<label>/`, reads
the compile-time parameters back from `out/out.json`, and writes
`runs/<label>_run<NN>/` with `device_final.npy` (final `[x,y,z,vx,vy,vz,m,
fx,fy,fz]`), `results.json` (per-step and aggregate cycle timings, throughput),
`run.log` and `run_meta.json`.

### 4. Validate

Because a gravitational N-body system is chaotic (fp32 rounding diverges
exponentially), correctness is **not** checked by comparing final trajectories
to a CPU reference. Instead the validator checks quantities that are invariant
to rounding order:

```bash
# invariants from a single run
python validation/validate_nbody.py \
       --bodies bodies/N4096_conv/bodies.npy \
       --final  runs/SIDE64_NL1_run01/device_final.npy

# invariants + order of convergence (several dt at the same physical time T)
python validation/validate_nbody.py \
       --bodies bodies/N4096_conv/bodies.npy \
       --final  runs/CONV_016/device_final.npy \
                runs/CONV_032/device_final.npy \
                runs/CONV_064/device_final.npy \
                runs/CONV_128/device_final.npy
```

A correct run shows: total **energy** drift falling ~4× per `dt` halving
(O(dt²)), **linear and angular momentum** drift small relative to their scales
(Newton's third law + central forces, i.e. the ring reduction is symmetric),
and a reference-free self-convergence order settling on **~2.0** (second-order
leapfrog). `dt`/`N_STEPS` are read automatically from each run's
`results.json`.

A separate workflow (`validation/n2_cpu.py`, a bit-order-faithful fp32 replica
of the kernel's ring reduction, plus `pe_validate.csl`) checks **one-step force
accuracy** pointwise against the CPU — mean relative error ~2e-6.

---

## Device-side exports

| symbol | type | content |
|---|---|---|
| `ptr_local_particles` | `f32` | per-PE bodies `[x,y,z,vx,vy,vz,m] × N_LOCAL` (in and out) |
| `ptr_local_forces` | `f32` | per-PE final forces `[fx,fy,fz] × N_LOCAL` |
| `ptr_done` | `u32[1]` | 1 when the PE has finished all steps |
| `ptr_n_pred_fwd` | `u32[1]` | chunks to receive on the FWD stream (= ring position `k`) |
| `ptr_n_pred_bwd` | `u32[1]` | chunks to receive on the BWD stream (= `side²-1-k`) |
| `ptr_time_buf_u16` | `u16` | 9×u16 timestamps per step × `(N_STEPS+1)` |
| `runtime_init` | fn | unblocks the command stream; launch first |
| `compute_step` | fn | runs all `N_STEPS` leapfrog steps; host polls `done` |
