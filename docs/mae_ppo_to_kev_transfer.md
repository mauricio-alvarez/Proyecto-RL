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

## Full-Environment MAE PPO Checkpoint

The same zero-shot transfer evaluation was also run with the MAE PPO checkpoint trained on:

```text
multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet
```

Full MAE checkpoint:

```text
runs/mae_ppo_full_v1/model.ckpt-500
runs/mae_ppo_full_v1/normalization.npz
runs/mae_ppo_full_v1/summary.json
```

Full training summary:

```text
updates: 500
episodes_completed: 1066
final deterministic MAE evaluation: hiders 4/5, seekers 1/5
mean_visible_fraction: 0.5092
```

KEV transfer output:

```text
runs/mae_ppo_full_kev_transfer_v1/report.md
runs/mae_ppo_full_kev_transfer_v1/summary.json
runs/mae_ppo_full_kev_transfer_v1/episodes.csv
runs/mae_ppo_full_kev_transfer_v1/pair_summaries.csv
```

Aggregate by attacker:

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| targeted | 0.300 | 0.700 | 0.000 | -28.25 |
| mae_ppo_transfer_attacker | 0.190 | 0.768 | 0.042 | -48.09 |
| linear_q_kev | 0.095 | 0.005 | 0.900 | -82.45 |
| stealth | 0.033 | 0.382 | 0.585 | -82.82 |
| greedy | 0.020 | 0.980 | 0.000 | -89.30 |
| random | 0.015 | 0.712 | 0.273 | -90.32 |
| q_table_kev | 0.000 | 0.163 | 0.837 | -95.69 |

The full-environment MAE transfer attacker improves over the quadrant transfer:

```text
quadrant MAE transfer success: 0.123
full MAE transfer success:     0.190
```

It also improves over Q-learning on attacker success:

```text
full MAE transfer: 0.190
linear_q_kev:      0.095
q_table_kev:       0.000
```

The same limitation remains: detection risk is high.

```text
full MAE transfer caught: 0.768
linear_q_kev caught:     0.005
```

Conclusion: training on `hide_and_seek_full` transferred a stronger cyber attacker signal than the quadrant checkpoint, but it still behaves aggressively rather than stealthily. The next implementation should fine-tune this transferred policy inside KEV with a detection-aware reward.

## Continued Full-Environment Checkpoint

The full-environment checkpoint was continued from update 500 to update 1000:

```text
runs/mae_ppo_full_v2_continue/model.ckpt-1000
runs/mae_ppo_full_v2_continue/normalization.npz
runs/mae_ppo_full_v2_continue/summary.json
```

MAE game evaluation at checkpoint 1000:

```text
episodes: 100
hider_wins: 56
seeker_wins: 43
ties: 1
hider_win_rate: 0.5600
seeker_win_rate: 0.4300
mean_visible_fraction: 0.5460
```

The checkpoint-1000 KEV transfer evaluation was run with:

```bash
docker run --rm -e PYTHONUNBUFFERED=1 -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_mae_ppo_kev_transfer.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --checkpoint /workspace/runs/mae_ppo_full_v2_continue/model.ckpt-1000 \
    --normalization /workspace/runs/mae_ppo_full_v2_continue/normalization.npz \
    --train-summary /workspace/runs/mae_ppo_full_v2_continue/summary.json \
    --q-table-json /workspace/runs/kev_realworld_benchmark_v1/q_table_kev.json \
    --linear-q-json /workspace/runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json \
    --out-dir /workspace/runs/mae_ppo_full_v2_kev_transfer_v1 \
    --episodes-per-family 20 \
    --seed 5400 \
    --n-nodes 8 \
    --max-steps 24
```

Output artifacts:

```text
runs/mae_ppo_full_v2_kev_transfer_v1/report.md
runs/mae_ppo_full_v2_kev_transfer_v1/summary.json
runs/mae_ppo_full_v2_kev_transfer_v1/episodes.csv
runs/mae_ppo_full_v2_kev_transfer_v1/pair_summaries.csv
```

Aggregate by attacker:

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| targeted | 0.258 | 0.742 | 0.000 | -38.03 |
| mae_ppo_transfer_attacker | 0.143 | 0.808 | 0.048 | -58.59 |
| linear_q_kev | 0.080 | 0.010 | 0.910 | -85.76 |
| stealth | 0.022 | 0.388 | 0.590 | -85.29 |
| random | 0.013 | 0.697 | 0.290 | -90.69 |
| greedy | 0.012 | 0.988 | 0.000 | -91.24 |
| q_table_kev | 0.003 | 0.105 | 0.892 | -95.38 |

Aggregate by defender:

| defender | attacker success | caught | timeout | return |
|---|---:|---:|---:|---:|
| adaptive | 0.034 | 0.683 | 0.283 | -90.50 |
| decoy_frontier | 0.037 | 0.730 | 0.233 | -91.46 |
| patch_high_value | 0.041 | 0.547 | 0.411 | -85.28 |
| mae_ppo_transfer_defender | 0.081 | 0.403 | 0.516 | -72.25 |
| random | 0.083 | 0.544 | 0.373 | -76.78 |
| noop | 0.179 | 0.297 | 0.524 | -50.84 |

Checkpoint comparison:

| source checkpoint | MAE env | MAE eval hider win | MAE eval seeker win | KEV transfer success | KEV transfer caught |
|---|---|---:|---:|---:|---:|
| `mae_ppo_quadrant_v1/model.ckpt-300` | quadrant | 0.460 | 0.520 | 0.123 | 0.847 |
| `mae_ppo_full_v1/model.ckpt-500` | full | 0.400 | 0.580 | 0.190 | 0.768 |
| `mae_ppo_full_v2_continue/model.ckpt-1000` | full | 0.560 | 0.430 | 0.143 | 0.808 |

Interpretation:

```text
More MAE training improved the hide-and-seek game balance, but it did not monotonically improve KEV transfer.
```

The best KEV transferred attacker so far is checkpoint 500 from `hide_and_seek_full`, not checkpoint 1000. The checkpoint-1000 model is better balanced in MAE, but the zero-shot adapter maps it into a less effective cyber attacker. This means the current bottleneck is not only MAE policy quality; it is also the transfer interface between physical hide-and-seek actions and cyber attack/defense actions.

Practical next step:

```text
Use the checkpoint-500 full MAE transfer as the initialization for KEV fine-tuning, then optimize directly on KEV with a detection-aware reward.
```

Do not spend large compute on more MAE-only updates before adding KEV fine-tuning. The evidence now shows that more source-domain training can improve source-domain performance while reducing target-domain transfer quality.

## KEV Fine-Tuning Result

The checkpoint-500 full-environment transfer was fine-tuned on KEV with:

```text
script: scripts/finetune_mae_ppo_kev.py
run_dir: runs/mae_ppo_full500_kev_finetune_v1
source_checkpoint: runs/mae_ppo_full_v1/model.ckpt-500
finetune_episodes: 500
mae_feature_q_episodes: 500
```

The useful fine-tuned model is:

```text
mae_ppo_kev_q_finetuned_attacker
```

Result:

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| mae_ppo_transfer_attacker | 0.197 | 0.747 | 0.056 | -45.35 |
| mae_ppo_kev_q_finetuned_attacker | 0.133 | 0.053 | 0.814 | -69.19 |
| linear_q_kev | 0.069 | 0.011 | 0.920 | -88.01 |
| q_table_kev | 0.001 | 0.121 | 0.877 | -95.30 |

Fine-tuning did not beat the zero-shot MAE attacker on raw success, but it substantially reduced caught rate while still outperforming both Q-learning baselines on success. This makes it the current best detection-aware transferred attacker.

Full details are in:

```text
docs/kev_finetuning.md
runs/mae_ppo_full500_kev_finetune_v1/report.md
```

## Success-First Targeted Distillation

The timeout-heavy behavior of the detection-aware fine-tuned model was addressed with a targeted-distilled MAE-feature Q attacker:

```text
mae_ppo_targeted_q_finetuned_attacker
```

This model uses:

```text
MAE checkpoint-500 policy statistics
targeted-baseline action preferences
KEV progress-shaped reward
```

Best current result:

```text
run_dir: runs/mae_ppo_success_first_50ep
```

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| mae_ppo_targeted_q_finetuned_attacker | 0.276 | 0.724 | 0.000 | -31.52 |
| targeted | 0.273 | 0.727 | 0.000 | -32.16 |
| mae_ppo_transfer_attacker | 0.157 | 0.803 | 0.040 | -52.71 |
| linear_q_kev | 0.063 | 0.017 | 0.920 | -89.25 |

This catches up to the targeted baseline on raw success. It does not solve stealth yet.

Longer success-first Q fine-tuning was also tested:

```text
run_dir: runs/mae_ppo_full500_kev_success_first_v1
success: 0.061
caught: 0.200
timeout: 0.739
```

Conclusion: targeted behavior can be transferred into the MAE-feature policy, but unrestricted 500-episode Q fine-tuning overcorrects toward timeout-heavy behavior. The next step is validation-based early stopping or constrained updates that preserve targeted success while reducing caught rate.

## Other MAE Source Environments

The remaining MAE example environments were trained as single-role seeker policies and evaluated in KEV:

```text
blueprint
shelter
lock_and_return
sequential_lock
```

Because these environments are single-agent tasks, they were trained with:

```text
--n-hiders-override 0
```

and evaluated with:

```text
--exclude-mae-defender
```

Same-seed KEV attacker-only comparison:

| source checkpoint | KEV success | caught | timeout | return |
|---|---:|---:|---:|---:|
| hide_and_seek_full_500 | 0.206 | 0.744 | 0.050 | -47.72 |
| hide_and_seek_full_1000 | 0.202 | 0.746 | 0.052 | -48.59 |
| sequential_lock | 0.186 | 0.786 | 0.028 | -51.64 |
| blueprint | 0.176 | 0.798 | 0.026 | -53.82 |
| lock_and_return | 0.160 | 0.828 | 0.012 | -57.05 |
| hide_and_seek_quadrant | 0.148 | 0.828 | 0.024 | -59.67 |
| shelter | 0.124 | 0.876 | 0.000 | -64.19 |

The full hide-and-seek checkpoint remains the best source model. The useful new finding is that `sequential_lock` and `blueprint` transfer better than quadrant under the attacker-only protocol, so they are reasonable auxiliary sources for future multi-source transfer.

Full report:

```text
docs/mae_other_envs_kev_transfer.md
```
