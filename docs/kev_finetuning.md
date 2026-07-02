# KEV Fine-Tuning From MAE PPO Checkpoint 500

This experiment fine-tunes cyber policies on the CISA KEV environment using the full-environment MAE PPO checkpoint at update 500 as the source model.

Source checkpoint:

```text
runs/mae_ppo_full_v1/model.ckpt-500
runs/mae_ppo_full_v1/normalization.npz
runs/mae_ppo_full_v1/summary.json
```

The MAE policy cannot be directly continued in KEV because MAE and KEV have different observation and action spaces. The implemented fine-tuning path keeps the MAE checkpoint as a source feature model:

```text
KEV state -> MAE-compatible concept vector -> MAE checkpoint policy stats -> KEV cyber policy
```

Two fine-tuned variants are produced:

- `mae_ppo_kev_finetuned_attacker` and `mae_ppo_kev_finetuned_defender`: neural cyber policies warm-started from checkpoint-500 transfer labels, then updated with policy-gradient rollouts.
- `mae_ppo_kev_q_finetuned_attacker`: a MAE-feature Q attacker initialized with checkpoint-500 transfer preferences, then trained directly on KEV reward.
- `mae_ppo_targeted_q_finetuned_attacker`: a success-first MAE-feature Q attacker initialized with targeted-baseline action preferences and MAE checkpoint-500 policy statistics.

The useful results are split:

- `mae_ppo_targeted_q_finetuned_attacker` closes the raw success gap to the handcrafted `targeted` baseline.
- `mae_ppo_kev_q_finetuned_attacker` is safer than zero-shot transfer, but times out heavily.

The neural policy-gradient attacker collapsed into mostly timeouts in this 500-episode run.

## Command

```bash
docker run --rm -e PYTHONUNBUFFERED=1 -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/finetune_mae_ppo_kev.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --checkpoint /workspace/runs/mae_ppo_full_v1/model.ckpt-500 \
    --normalization /workspace/runs/mae_ppo_full_v1/normalization.npz \
    --train-summary /workspace/runs/mae_ppo_full_v1/summary.json \
    --q-table-json /workspace/runs/kev_realworld_benchmark_v1/q_table_kev.json \
    --linear-q-json /workspace/runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json \
    --out-dir /workspace/runs/mae_ppo_full500_kev_finetune_v1 \
    --episodes 500 \
    --mae-q-episodes 500 \
    --bc-samples 4000 \
    --bc-epochs 8 \
    --batch-size 512 \
    --hidden-sizes 192,192 \
    --learning-rate 0.0007 \
    --episodes-per-family 20 \
    --seed 6200 \
    --n-nodes 8 \
    --max-steps 24
```

Runtime on the local Docker setup was about 7 minutes.

## Artifacts

```text
runs/mae_ppo_full500_kev_finetune_v1/report.md
runs/mae_ppo_full500_kev_finetune_v1/summary.json
runs/mae_ppo_full500_kev_finetune_v1/progress.csv
runs/mae_ppo_full500_kev_finetune_v1/finetune_progress.svg
runs/mae_ppo_full500_kev_finetune_v1/mae_feature_q_attacker.json
runs/mae_ppo_full500_kev_finetune_v1/attacker_policy.ckpt*
runs/mae_ppo_full500_kev_finetune_v1/defender_policy.ckpt*
```

## Result

Evaluation used:

```text
families: recent_enterprise, ransomware_focus, vendor_cluster, cwe_cluster, mixed_kev
episodes_per_family: 20
n_nodes: 8
max_steps: 24
```

Aggregate by attacker:

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| targeted | 0.289 | 0.711 | 0.000 | -30.25 |
| mae_ppo_transfer_attacker | 0.197 | 0.747 | 0.056 | -45.35 |
| mae_ppo_kev_q_finetuned_attacker | 0.133 | 0.053 | 0.814 | -69.19 |
| linear_q_kev | 0.069 | 0.011 | 0.920 | -88.01 |
| stealth | 0.033 | 0.323 | 0.644 | -81.03 |
| random | 0.019 | 0.639 | 0.343 | -88.41 |
| greedy | 0.016 | 0.984 | 0.000 | -89.16 |
| q_table_kev | 0.001 | 0.121 | 0.877 | -95.30 |
| mae_ppo_kev_finetuned_attacker | 0.000 | 0.193 | 0.807 | -92.82 |

Aggregate by defender:

| defender | attacker success | caught | timeout | return |
|---|---:|---:|---:|---:|
| adaptive | 0.036 | 0.604 | 0.360 | -89.37 |
| decoy_frontier | 0.038 | 0.637 | 0.326 | -91.04 |
| patch_high_value | 0.047 | 0.440 | 0.513 | -84.38 |
| mae_ppo_transfer_defender | 0.093 | 0.310 | 0.597 | -70.13 |
| mae_ppo_kev_finetuned_defender | 0.097 | 0.302 | 0.601 | -69.33 |
| random | 0.099 | 0.433 | 0.468 | -72.85 |
| noop | 0.179 | 0.216 | 0.606 | -51.42 |

## Interpretation

The fine-tuned Q attacker improves over both Q-learning baselines on attacker success:

```text
mae_ppo_kev_q_finetuned_attacker: 0.133
linear_q_kev:                     0.069
q_table_kev:                      0.001
```

It also reduces detection substantially compared with the zero-shot MAE transfer attacker:

```text
mae_ppo_transfer_attacker caught:        0.747
mae_ppo_kev_q_finetuned_attacker caught: 0.053
```

The tradeoff is that the fine-tuned Q attacker times out more often:

```text
mae_ppo_kev_q_finetuned_attacker timeout: 0.814
mae_ppo_transfer_attacker timeout:        0.056
```

So the current fine-tuning succeeded at learning a safer KEV attacker from MAE transfer features, but it did not yet produce the best raw success rate. The next optimization should explicitly target the success/stealth frontier: preserve the zero-shot MAE attack progression when detection is low, and switch to the fine-tuned Q behavior when detection risk is high.

## Success-First Targeted Distillation

To address the timeout failure mode, a success-first variant was added to `scripts/finetune_mae_ppo_kev.py`. It uses:

```text
MAE checkpoint-500 policy statistics
targeted-baseline action imitation
progress-shaped KEV reward
```

The key run is:

```text
runs/mae_ppo_success_first_50ep
```

Command:

```bash
docker run --rm -e PYTHONUNBUFFERED=1 -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/finetune_mae_ppo_kev.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --checkpoint /workspace/runs/mae_ppo_full_v1/model.ckpt-500 \
    --normalization /workspace/runs/mae_ppo_full_v1/normalization.npz \
    --train-summary /workspace/runs/mae_ppo_full_v1/summary.json \
    --q-table-json /workspace/runs/kev_realworld_benchmark_v1/q_table_kev.json \
    --linear-q-json /workspace/runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json \
    --out-dir /workspace/runs/mae_ppo_success_first_50ep \
    --episodes 0 \
    --mae-q-episodes 0 \
    --success-q-episodes 50 \
    --bc-samples 64 \
    --bc-epochs 1 \
    --batch-size 32 \
    --hidden-sizes 32 \
    --success-q-alpha 0.002 \
    --success-q-epsilon-start 0.12 \
    --success-q-epsilon-end 0.02 \
    --episodes-per-family 20 \
    --seed 6310 \
    --n-nodes 8 \
    --max-steps 24
```

Result on the same held-out KEV evaluation seeds:

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| mae_ppo_targeted_q_finetuned_attacker | 0.276 | 0.724 | 0.000 | -31.52 |
| targeted | 0.273 | 0.727 | 0.000 | -32.16 |
| mae_ppo_transfer_attacker | 0.157 | 0.803 | 0.040 | -52.71 |
| linear_q_kev | 0.063 | 0.017 | 0.920 | -89.25 |
| q_table_kev | 0.014 | 0.111 | 0.874 | -92.40 |

This catches up to the targeted baseline on success. It does not yet improve the caught rate, which means the targeted behavior has been transferred but stealth has not been solved.

A 500-episode shaped Q update was also tested:

```text
runs/mae_ppo_full500_kev_success_first_v1
```

Result:

| attacker | success | caught | timeout | return |
|---|---:|---:|---:|---:|
| targeted | 0.273 | 0.727 | 0.000 | -32.16 |
| mae_ppo_targeted_q_finetuned_attacker | 0.061 | 0.200 | 0.739 | -82.49 |

The 500-episode update reduces caught rate, but destroys success by reintroducing the timeout problem. The current best success-first model is therefore the targeted-distilled initialization or very short update, not the long Q update.

Next implementation requirement:

```text
Add validation-based early stopping for success-first Q fine-tuning.
```

The stopping score should prioritize success first and caught rate second:

```text
score = success_rate - 0.35 * caught_rate - 0.20 * timeout_rate
```

This prevents the model from drifting into safe-but-unsuccessful timeout behavior.
