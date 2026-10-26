# Local probe replay performance

Reusing identical prefixes reduced the observed median complete replay time by
**9.84%**, from 585.579 to 527.982 seconds, in one A/B/B/A set on the local RTX 4090.
Both adjacent comparisons improved, by 19.01% and 1.27%. All scoring outputs matched
exactly. The result applies to the 336 saved contexts and conditions on the shared host below.

The local scorer repeats a complete CPU-offloaded model forward when different saved
requests deliver identical prefix tokens and measured token positions. The candidate
reuses only the five projected role means within one validated model/probe invocation.
It still computes each request's decision, thresholds, mask count, fingerprint, and
durable cache record. Model, checkpoint, runtime, and probe validation remain in force.

## Workload and identities

This experiment replays every available context in the existing
`pilot-nemotron-no-probe-20260916/collection` corpus. It adds no synthetic contexts or
copies to increase the number of repeated inputs.

| Item | Value |
|---|---|
| Baseline commit | `aacf59798c4787eea92e5be627197923e13ad330` |
| Candidate code commit | `fef3d2c2c0ec75c60cca2781c8701a3cd1d51b36` |
| Candidate `reasonese/` Git tree | `16bdf7835614393732f1d5e6ef7d5b99bc84956c` |
| Study specifications | 720 |
| Studies with saved traces | 85 |
| Scoreable requests | 336 |
| Distinct prefix/position inputs | 232 |
| Repeated captures | 104 |
| Missing contexts | 2,544 of 2,880 planned requests |
| Covered axes | All three channels and all eight framings |
| Local device | NVIDIA RTX 4090, 24,564 MiB, PCIe Gen 4 x16 |
| Host CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads |
| Runtime | PyTorch 2.13.0+cu130, CUDA 13.0, driver 580.82.09 |
| Probe | Frozen standardized Nemotron layer 13 |
| Probe SHA-256 | `011e2642cb4dd8741da7cc95a637780463b5b57a5c0842f368f9242d027d449c` |
| Complete checkpoint identity | `68286acc726d8f4566c99fd22a5dfee4a4557ca305af0a1c690efbf4681ee4d0` |

The complete original prefix-26 checkpoint is verified, and production scoring loads
the model only through the selected layer 13. The pinned H1 cumsum and CPU-offload
policy are unchanged. The source pilot's incomplete coverage and six reference
mismatches are retained; this optimization does not establish prompt adherence or
hosted-model parity.

## Baseline profile

The baseline completed all 336 requests with zero scoring errors and unchanged source
YAML/database hashes. All 104 repeated captures had bitwise-identical activations.

The complete CPU profile recorded 768.971 seconds of self time. `Tensor.to` accounted
for 496.738 seconds (64.60%). File reads accounted for 159.835 seconds, and SHA-256
updates for 39.756 seconds. Three complete checkpoint identity checks took 196.678
cumulative seconds, overlapping those file-read/hash entries. Role projection took
0.253 cumulative seconds.

After the separate two-request warmup, the first capture in the freshly loaded scorer
transferred 14,237,871,296 bytes from host memory to the GPU in 103 events. Copy events
summed to 3.432 seconds; kernel events summed to 25.828 milliseconds. These event sums
describe that capture, including CPU offload transfers, and are not a replay latency
measurement.

The combined CPU/CUDA instrumentation left incomplete cumulative call-stack entries
for outer replay/capture wrappers. The attribution above uses flat CPU self times and
the CUDA event trace. The reusable runner below collects CPU profiles without nesting
a CUDA profiler, and retains activation digests in profiling runs.

These observations support testing whether avoiding 104 repeated forwards reduces the
complete replay time. Profiled durations are excluded from the speed comparison.

## Fresh GPU correctness

The candidate completed all 336 requests with zero errors and unchanged source hashes.
It performed 232 captures instead of 336. Every distinct capture matched the baseline
input identity, shape, dtype, and activation bytes exactly.

The complete cache, report, and manifest comparison checked 4,033 floating-point
values: all were equal, with maximum absolute error zero. All discrete fields matched,
including decisions, masks, fingerprints, missing/error counts, and the six reference
mismatches. Only the three declared output-directory references were normalized.
The cache files were byte-identical (SHA-256
`2e8eacc98ad51d59b2136c074633ffbf7e709d4421745243d564b9e901759f73`).

Both profiled runs reached the same peak allocated CUDA memory, 2,728,494,592 bytes.
The candidate profile used the timing runner's earlier CUDA-context initialization;
an exact source comparison verified that this was the only runner change and that it
occurred before the fixed warmup and measured API. All four timing arms use the same
runner. Profiling durations are not compared as performance measurements.

## Qualification protocol

Each arm runs in a separate process with an empty output/cache directory. The same
interpreter, source collection, probe, checkpoint, thread settings, and owned compiler
caches are used throughout. Each process performs the same separate warmup on the
first and last real requests.

The timer surrounds the complete `score_saved_collections` call, including source
loading, preflight, integrity hashes, model loads, rendering, captures, projection,
cache persistence, and report writing. Imports, initial CUDA context setup, and the
separate warmup are excluded. CUDA synchronization brackets the measurement.

An idle GPU and at least 20 GiB available host RAM are required before admission. The
experiment supervisor stops only its own process if another GPU compute job appears;
interrupted attempts remain recorded. GPU/driver/PCIe state, host memory, load, and
disk counters are retained at timing-arm boundaries.

This is one A/B/B/A run set, with two observations per implementation. The host also
runs background CPU and disk workloads. The report retains every raw timing and both
adjacent comparisons; it does not estimate a confidence interval or a general speedup
distribution.

| Arm | Implementation | Complete replay seconds | Host one-minute load, start to end |
|---|---|---|---|
| A1 | Baseline | 565.581 | 8.76 to 11.08 |
| B1 | Candidate | 458.091 | 11.08 to 11.27 |
| B2 | Candidate | 597.872 | 11.27 to 19.68 |
| A2 | Baseline | 605.578 | 19.68 to 9.34 |

A separate observation during B2 recorded a one-minute host load of 23.32 and
12.37 GiB of available RAM. These observations describe host conditions and do not
attribute the timing variation to a single cause.

The A1/B1 pair reduced wall time by 19.005%; the A2/B2 pair reduced it by 1.273%.
The baseline median was 585.579 seconds and the candidate median was 527.982 seconds:
a 57.598-second reduction, or 9.836% (1.109x speedup). This passed the predeclared
requirement for improvement in both adjacent pairs and the medians. The large
variation between candidate runs limits the precision of this estimate from one run set.

Every unprofiled arm also passed the complete output comparison above, with
byte-identical caches, all 4,033 floating-point comparisons exactly equal, unchanged
source hashes, and the same peak GPU allocation of 2,728,494,592 bytes. Each arm
completed all 336 requests with zero errors and retained all 2,544 missing contexts.

Both adjacent comparisons (A1/B1 and A2/B2) and the arm medians must improve. Correctness
requires exact structure and discrete values in every cache record, report, and
manifest; finite floating-point values must satisfy `atol=rtol=1e-6`. Profiling also
compares every distinct capture's input identity, shape, dtype, and activation hash.
Source files and databases must remain unchanged.

## Reproduction

Install the locked environment with `uv sync --locked --extra probes`. Supply an
existing saved collection and a probe-bundle config pointing to the exact local
checkpoint and qualified probe. The helper accepts one bundle per experiment.

[`scripts/benchmark_probe_replay.py`](../../scripts/benchmark_probe_replay.py) uses the
same timed API call sequence and source inventory as the measurement runner. It writes
`launch.json`, `result.json`, a separate warmup cache, and the complete replay outputs.
Use `--cpu-profile` for a separate attribution run with CPU statistics and per-capture
activation digests; do not use those durations as speed measurements.

For a baseline or candidate checkout, select the production source tree with
`PYTHONPATH` while keeping the interpreter and benchmark script fixed:

```bash
PYTHONPATH="$probe_source_tree" "$probe_python" scripts/benchmark_probe_replay.py \
  --collection "$probe_collection" --role-probes "$probe_bundle" \
  --output "$probe_run/A1"
```

Run fresh directories in A1, B1, B2, A2 order during an uncontended GPU window. Set the
same `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` for all arms. Check the launch
records for matching source hashes, model/probe identities, runtime, and thread
settings before interpreting timings.

[`scripts/compare_probe_replays.py`](../../scripts/compare_probe_replays.py) compares
all three replay JSON artifacts and, when present in both runs, every activation
digest. Compare A1 with each other arm and retain all receipts:

```bash
"$probe_python" scripts/compare_probe_replays.py "$probe_run/A1" "$probe_run/B1" \
  --output "$probe_run/B1-parity.json"
```

The comparator rejects missing records, changed discrete values, nonfinite or
out-of-tolerance numbers, unexpected output paths, mismatched activation digests,
one-sided activation evidence, and an existing output receipt. Its checks also run
under Python's `-O` mode. All generated evidence belongs below ignored `out/`.

These results apply to local diagnostic replay over the stated saved
contexts. They do not measure remote collection throughput or establish performance
on other models, corpora, or GPU configurations. The required offline gate validates
the implementation; measured parity and timing are separate evidence.
