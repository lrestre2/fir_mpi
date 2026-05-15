# Parallel FIR Filter Architectures Using MPI
**CPSC-375 — High-Performance Computing, Spring 2026**
Liu Neptali Restrepo Sanabria — Trinity College, Hartford

---

## Overview

This project implements and benchmarks three FIR filter architectures in
distributed-memory MPI, directly analogous to the sequential, semi-parallel,
and fully parallel FPGA implementations described in the Xilinx XtremeDSP
design guide:

| Implementation | File | MPI Strategy |
|---|---|---|
| Sequential (baseline) | `src/fir_serial.c` | Single process, direct MAC loop |
| Semi-parallel | `src/fir_semi_parallel.c` | Coefficient decomposition + `MPI_Reduce` |
| Parallel | `src/fir_parallel.c` | Signal decomposition + halo exchange |

Shared utilities (coefficient generation, signal generation, verification)
are in `src/fir_common.h`.

---

## Directory Structure

```
fir_mpi/
├── README.md
├── Makefile
├── src/
│   ├── fir_common.h          # Shared header: coefficients, signal, verify
│   ├── fir_serial.c          # Sequential MAC FIR reference
│   ├── fir_semi_parallel.c   # Coefficient-decomposition MPI
│   └── fir_parallel.c        # Data-decomposition MPI with halo exchange
└── slurm/
    ├── run_serial.sbatch      # SLURM job: serial baseline sweep
    ├── run_semi.sbatch        # SLURM job: semi-parallel strong scaling
    └── run_parallel.sbatch    # SLURM job: parallel strong scaling
```

---

## Requirements

- OpenMPI 4.1.8 (or compatible)
- GCC with C99 support
- SLURM workload manager (for cluster runs)
- GNU Make

On the Pine cluster, load the required module before doing anything:

```bash
module load openmpi/gcc/64/4.1.8
```

---

## Building

From the project root directory:

```bash
make
```

This produces three binaries in the project root:

```
fir_serial
fir_semi_parallel
fir_parallel
```

To remove all binaries and reference files:

```bash
make clean
```

---

## Running

All three programs take the same two arguments:

```
./<binary> <N_taps> <L_samples>
```

- `N_taps` — number of FIR filter coefficients (e.g. 128)
- `L_samples` — length of the input signal (e.g. 10000000)

### Quick sanity test (interactive, before submitting jobs)

Always run the serial implementation first — it generates the binary
reference file used for correctness verification by the parallel runs.

```bash
# Step 1: generate reference output
mpirun -n 1 ./fir_serial 128 10000000

# Step 2: test semi-parallel against reference
mpirun -n 4 ./fir_semi_parallel 128 10000000

# Step 3: test parallel against reference
mpirun -n 4 ./fir_parallel 128 10000000
```

The parallel implementations print `max_abs_error` to stderr. Any value
below `1e-9` confirms correctness. The parallel implementation should
report exactly `0.000e+00`; the semi-parallel will report a small value
around `2e-15` due to floating-point reordering in `MPI_Reduce` — both
are correct.

### Output format

Each program prints one line to stdout per run:

```
<N>  <L>  [<P>]  <wall_time_seconds>
```

The serial implementation omits `P` (always 1). Example:

```
128 10000000 1.162652          # serial
128 10000000 8 0.166158        # parallel or semi, P=8
```

---

## Benchmarking on Pine

Submit all three jobs from the project root:

```bash
sbatch slurm/run_serial.sbatch
sbatch slurm/run_semi.sbatch
sbatch slurm/run_parallel.sbatch
```

Monitor job status:

```bash
squeue -u <yournetid>
```

Watch output live:

```bash
tail -f logs/serial_<jobid>.out
tail -f logs/semi_<jobid>.out
tail -f logs/parallel_<jobid>.out
```

Output and error files are written to `logs/` (created automatically by
the job scripts). **Always run the serial job first** to generate the
reference `.bin` files before submitting the parallel jobs, otherwise
verification will be skipped.

### What each job does

| Script | Nodes | Tasks | Experiment |
|---|---|---|---|
| `run_serial.sbatch` | 1 | 1 | Baseline timing at N=128, L=10^7; filter length sweep (N=32–512, L=10^6) |
| `run_semi.sbatch` | 8 | 192 | Strong scaling P=1–192 at N=128, L=10^7; filter length sweep at P=8 |
| `run_parallel.sbatch` | 8 | 192 | Strong scaling P=1–192 at N=128, L=10^7; filter length sweep at P=8 |

Expected runtimes: serial ~2 min, semi-parallel ~5 min, parallel ~5 min.

### Note on SLURM warnings

For small process counts (P=1, 2, 4), SLURM will print warnings such as:

```
srun: warning: can't run 1 processes on 8 nodes, setting nnodes to 1
```

This is expected. SLURM automatically reduces the active node count to
match the process count for that step. It does not affect correctness
or timing.

---

## Reference Files

The serial implementation saves its output to a binary file:

```
ref_N<N>_L<L>.bin
```

For example, `ref_N128_L10000000.bin`. The parallel implementations load
this file automatically and print the maximum absolute error against it.
If no reference file is found, verification is silently skipped.

---

## Computing Metrics from Results

Given the output files, speedup, efficiency, and Karp--Flatt are computed as:

```
S(P) = T1 / T_P
E(P) = S(P) / P * 100%
e(P) = (1/S(P) - 1/P) / (1 - 1/P)
```

where `T1` is the wall-clock time from `fir_serial` at N=128, L=10^7.