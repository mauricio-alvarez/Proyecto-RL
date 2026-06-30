# Real-Data Cyber Benchmark: CISA KEV

This pipeline connects the cyber hide-and-seek environment to a real cybersecurity dataset: CISA's Known Exploited Vulnerabilities catalog.

Source:

- `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`

Local dataset:

- `data/raw/known_exploited_vulnerabilities.json`

Validated catalog metadata:

```text
title: CISA Catalog of Known Exploited Vulnerabilities
catalogVersion: 2026.06.25
dateReleased: 2026-06-25T19:03:21.8037Z
count: 1629
```

The catalog contains real CVE, vendor, product, CWE, ransomware-use, date-added, and required-action metadata. The project does not use exploit code, does not scan live systems, and does not target real assets.

## Implementation

Implemented files:

- `scripts/fetch_cisa_kev.py`
- `cyber_rl/kev.py`
- `scripts/run_kev_benchmarks.py`

The adapter maps KEV records into safe abstract cyber scenarios:

- each node is backed by one real KEV record
- node metadata includes CVE, vendor, product, vulnerability name, CWE, and ransomware-use flag
- exploit difficulty is a bounded heuristic risk score from CVE metadata
- target value is derived from risk/ransomware/recency heuristics
- topology is synthetic but parameterized by real vulnerability clusters

KEV scenario families:

- `recent_enterprise`: recent exploited vulnerabilities arranged as enterprise progression.
- `ransomware_focus`: vulnerabilities marked as known ransomware campaign use.
- `vendor_cluster`: records clustered by vendor/project.
- `cwe_cluster`: records clustered by weakness class.
- `mixed_kev`: random KEV-derived topology.

This creates a real-data-backed benchmark world without creating operational attack tooling.

## Reproduce

Download the dataset:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/fetch_cisa_kev.py \
    --out /workspace/data/raw/known_exploited_vulnerabilities.json
```

Run a small smoke benchmark:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/run_kev_benchmarks.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --out-dir /workspace/runs/kev_realworld_smoke \
    --episodes-per-family 3 \
    --q-train-episodes 20 \
    --linear-q-train-episodes 30 \
    --seed 2100 \
    --n-nodes 8 \
    --max-steps 24
```

Validated smoke result:

```text
episodes: 450
pair_summaries: 30
```

Run the full real-data benchmark:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/run_kev_benchmarks.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --out-dir /workspace/runs/kev_realworld_benchmark_v1 \
    --episodes-per-family 50 \
    --q-train-episodes 8000 \
    --linear-q-train-episodes 14000 \
    --seed 2200 \
    --n-nodes 8 \
    --max-steps 24
```

## Outputs

Main output directory:

- `runs/kev_realworld_benchmark_v1/`

Files:

- `report.md`: human-readable result table
- `summary.json`: full benchmark result
- `episodes.csv`: per-episode result table
- `pair_summaries.csv`: attacker/defender matchup table
- `attacker_success.svg`: plot of aggregate seeker success
- `kev_dataset_summary.json`: KEV catalog metadata
- `kev_scenario_samples.json`: sample scenarios with node-level CVE metadata
- `q_table_kev_training_summary.json`: tabular Q training summary
- `q_table_kev.json`: trained tabular Q table
- `linear_q_kev_training_summary.json`: linear Q training summary
- `linear_q_kev_weights.json`: trained linear Q weights

## Validated Result

Full benchmark size:

```text
episodes: 7500
pair_summaries: 30
```

Aggregate seeker results:

| seeker | success | caught | timeout | mean return |
|---|---:|---:|---:|---:|
| targeted | 0.281 | 0.719 | 0.000 | -34.02 |
| linear_q_kev | 0.126 | 0.006 | 0.868 | -75.85 |
| stealth | 0.047 | 0.442 | 0.510 | -81.14 |
| greedy | 0.027 | 0.973 | 0.000 | -88.06 |
| random | 0.018 | 0.795 | 0.187 | -91.90 |
| q_table_kev | 0.007 | 0.198 | 0.795 | -95.27 |

Aggregate hider results:

| hider | attacker success | caught | timeout | mean attacker return |
|---|---:|---:|---:|---:|
| decoy_frontier | 0.049 | 0.697 | 0.255 | -89.11 |
| adaptive | 0.054 | 0.631 | 0.315 | -85.93 |
| patch_high_value | 0.061 | 0.489 | 0.450 | -82.13 |
| random | 0.097 | 0.510 | 0.393 | -74.67 |
| noop | 0.161 | 0.284 | 0.555 | -56.70 |

Interpretation:

- `targeted` remains the strongest seeker because it uses explicit target-progress structure.
- `linear_q_kev` is the strongest learned seeker, clearly above random, greedy, stealth, and tabular Q on success, and it is rarely caught.
- `q_table_kev` performs poorly because tabular state memorization does not generalize well across changing CVE-backed graph scenarios.
- `decoy_frontier` is the strongest hider in this real-data benchmark.

## Project Status

The project now has a complete real-data evaluation pipeline:

1. Fetch real vulnerability dataset.
2. Convert CVE records into safe benchmark scenarios.
3. Train Q-based seekers on KEV-derived worlds.
4. Evaluate learned and principle-based seekers against multiple hiders.
5. Save reproducible reports, CSVs, JSON, plot, and sample scenario traces.

The next model upgrade should be graph-aware neural RL. The current pipeline is ready to support it because the benchmark world and evaluation matrix are already implemented.
