# MAE PPO To KEV Transfer Evaluation

This experiment evaluates the trained MAE PPO hide-and-seek checkpoint on the CISA KEV cyber benchmark through a zero-shot concept adapter.

The MAE checkpoint cannot act directly in the cyber environment because the observation and action spaces are different:

- MAE uses physical observations and movement/grab/lock actions.
- KEV uses cyber graph state and scan/exploit/move/exfiltrate/patch/decoy/monitor actions.

The adapter maps KEV cyber state into a fixed MAE-compatible concept vector, queries the trained MAE hider/seeker policy logits, then translates:

```text
MAE seeker preferences -> cyber attacker action scores
MAE hider preferences  -> cyber defender action scores
```

This is zero-shot transfer. No KEV fine-tuning was applied to the MAE PPO checkpoint.

## Inputs

MAE PPO checkpoint:

```text
runs/mae_ppo_quadrant_v1/model.ckpt-300
runs/mae_ppo_quadrant_v1/normalization.npz
runs/mae_ppo_quadrant_v1/summary.json
```

KEV dataset:

```text
data/raw/known_exploited_vulnerabilities.json
```

Q-learning baselines:

```text
runs/kev_realworld_benchmark_v1/q_table_kev.json
runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json
```

## Reproduce

```bash
docker run --rm -e PYTHONUNBUFFERED=1 -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_mae_ppo_kev_transfer.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --checkpoint /workspace/runs/mae_ppo_quadrant_v1/model.ckpt-300 \
    --normalization /workspace/runs/mae_ppo_quadrant_v1/normalization.npz \
    --train-summary /workspace/runs/mae_ppo_quadrant_v1/summary.json \
    --q-table-json /workspace/runs/kev_realworld_benchmark_v1/q_table_kev.json \
    --linear-q-json /workspace/runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json \
    --out-dir /workspace/runs/mae_ppo_kev_transfer_v1 \
    --episodes-per-family 20 \
    --seed 5200 \
    --n-nodes 8 \
    --max-steps 24
```

## Outputs

```text
runs/mae_ppo_kev_transfer_v1/report.md
runs/mae_ppo_kev_transfer_v1/summary.json
runs/mae_ppo_kev_transfer_v1/episodes.csv
runs/mae_ppo_kev_transfer_v1/pair_summaries.csv
```

## Result

Evaluation size:

```text
episodes: 4200
families: recent_enterprise, ransomware_focus, vendor_cluster, cwe_cluster, mixed_kev
episodes_per_family: 20
```

Aggregate by attacker:

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| targeted | 0.260 | 0.740 | 0.000 | -38.76 |
| mae_ppo_transfer_attacker | 0.123 | 0.847 | 0.030 | -61.68 |
| linear_q_kev | 0.090 | 0.010 | 0.900 | -83.68 |
| stealth | 0.033 | 0.395 | 0.572 | -82.50 |
| greedy | 0.018 | 0.982 | 0.000 | -89.05 |
| random | 0.012 | 0.733 | 0.255 | -92.11 |
| q_table_kev | 0.007 | 0.195 | 0.798 | -94.85 |

Aggregate by defender:

| defender | attacker success | caught | timeout | return |
|---|---:|---:|---:|---:|
| decoy_frontier | 0.030 | 0.754 | 0.216 | -92.83 |
| adaptive | 0.044 | 0.689 | 0.267 | -87.76 |
| patch_high_value | 0.053 | 0.543 | 0.404 | -83.04 |
| random | 0.083 | 0.596 | 0.321 | -77.17 |
| mae_ppo_transfer_defender | 0.093 | 0.417 | 0.490 | -69.80 |
| noop | 0.163 | 0.346 | 0.491 | -54.52 |

## Interpretation

The MAE transfer attacker beats both Q-learning baselines on attacker success:

```text
mae_ppo_transfer_attacker: 0.123
linear_q_kev:             0.090
q_table_kev:              0.007
```

The tradeoff is detection risk:

```text
mae_ppo_transfer_attacker caught: 0.847
linear_q_kev caught:             0.010
```

So the MAE seeker transfer learned a more aggressive progression strategy than Q-learning, but it is not stealthy. The linear Q model remains the safer attacker because it avoids detection, but it times out in 90% of episodes.

The MAE transfer defender is weaker than the best handcrafted defenders:

```text
decoy_frontier attacker success:        0.030
adaptive attacker success:              0.044
mae_ppo_transfer_defender success:      0.093
```

This is expected for zero-shot transfer. The next step should fine-tune the MAE-derived cyber attacker/defender on KEV trajectories rather than only using a fixed adapter.
