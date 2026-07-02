# MAE Other-Environments To KEV Transfer

This experiment trains the remaining MAE example environments and evaluates their transferred seeker policy in the KEV cyber benchmark.

The original project examples are:

```text
hide_and_seek_quadrant.jsonnet
hide_and_seek_full.jsonnet
blueprint.jsonnet
shelter.jsonnet
lock_and_return.jsonnet
sequential_lock.jsonnet
```

`hide_and_seek_quadrant` and `hide_and_seek_full` are native two-role hider/seeker environments. The other four are single-agent construction or box-locking environments. For those, the MAE PPO trainer was run with:

```text
--n-hiders-override 0
```

That trains the single agent through the MAE `seeker` policy scope, making the checkpoint usable as a transferred KEV attacker. During KEV evaluation, the MAE defender was excluded:

```text
--exclude-mae-defender
```

This keeps the comparison focused on the trained seeker/attacker policy instead of including an untrained hider/defender scope.

## Training Command

Each new environment was trained for 200 PPO updates:

```bash
docker run --rm -e PYTHONUNBUFFERED=1 -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_mae_ppo.py \
    --env /workspace/multi-agent-emergence-environments/examples/<env>.jsonnet \
    --out-dir /workspace/runs/mae_ppo_<env>_seeker_v1 \
    --updates 200 \
    --rollout-steps 256 \
    --ppo-epochs 3 \
    --batch-size 256 \
    --hidden-sizes 128,128 \
    --learning-rate 0.0003 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --entropy-coef 0.01 \
    --eval-every 50 \
    --eval-episodes 3 \
    --save-every 50 \
    --seed 7201 \
    --n-hiders-override 0
```

Trained checkpoints:

```text
runs/mae_ppo_blueprint_seeker_v1/model.ckpt-200
runs/mae_ppo_shelter_seeker_v1/model.ckpt-200
runs/mae_ppo_lock_and_return_seeker_v1/model.ckpt-200
runs/mae_ppo_sequential_lock_seeker_v1/model.ckpt-200
```

## KEV Evaluation Command

```bash
docker run --rm -e PYTHONUNBUFFERED=1 -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_mae_ppo_kev_transfer.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --checkpoint /workspace/runs/mae_ppo_<env>_seeker_v1/model.ckpt-200 \
    --normalization /workspace/runs/mae_ppo_<env>_seeker_v1/normalization.npz \
    --train-summary /workspace/runs/mae_ppo_<env>_seeker_v1/summary.json \
    --q-table-json /workspace/runs/kev_realworld_benchmark_v1/q_table_kev.json \
    --linear-q-json /workspace/runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json \
    --out-dir /workspace/runs/mae_ppo_<env>_seeker_kev_transfer_v1 \
    --episodes-per-family 20 \
    --seed 7300 \
    --n-nodes 8 \
    --max-steps 24 \
    --exclude-mae-defender
```

Existing hide-and-seek checkpoints were also reevaluated with the same attacker-only KEV protocol and seed for a fair comparison.

## Source Training Summary

| source env | updates | episodes | recent source return |
|---|---:|---:|---:|
| blueprint | 200 | 341 | -11.1157 |
| shelter | 200 | 213 | -12.8796 |
| lock_and_return | 200 | 426 | 2.0455 |
| sequential_lock | 200 | 426 | 1.1956 |

`lock_and_return` and `sequential_lock` learned positive source-task return. `blueprint` and `shelter` completed training but still had negative recent source-task return.

## KEV Transfer Result

All rows below use the same KEV evaluation seed and exclude the MAE transfer defender.

| source checkpoint | KEV success | caught | timeout | return |
|---|---:|---:|---:|---:|
| hide_and_seek_full_500 | 0.206 | 0.744 | 0.050 | -47.72 |
| hide_and_seek_full_1000 | 0.202 | 0.746 | 0.052 | -48.59 |
| sequential_lock | 0.186 | 0.786 | 0.028 | -51.64 |
| blueprint | 0.176 | 0.798 | 0.026 | -53.82 |
| lock_and_return | 0.160 | 0.828 | 0.012 | -57.05 |
| hide_and_seek_quadrant | 0.148 | 0.828 | 0.024 | -59.67 |
| shelter | 0.124 | 0.876 | 0.000 | -64.19 |

Baseline attackers on the same KEV protocol:

```text
targeted success:     0.300
linear_q_kev success: 0.126
q_table_kev success:  0.012
```

## Interpretation

The full hide-and-seek checkpoint remains the best MAE source environment for KEV attacker transfer.

The new result is still useful: `sequential_lock` and `blueprint` transfer better than `hide_and_seek_quadrant` under the attacker-only KEV protocol.

```text
sequential_lock:        0.186
blueprint:              0.176
hide_and_seek_quadrant: 0.148
```

This suggests that non-adversarial MAE skills involving object manipulation, locking, and construction can transfer some useful action preference into KEV. However, they do not beat `hide_and_seek_full`, which remains the most relevant source task because it trains adversarial pursuit in a multi-agent setting.

Practical conclusion:

```text
Keep hide_and_seek_full as the main source model.
Use sequential_lock and blueprint as auxiliary pretraining candidates if we implement multi-source transfer or policy ensembles.
Do not prioritize shelter for KEV attacker transfer.
```
