# MAE PPO-GAE Self-Play Training

This is the main training path for learning hider and seeker behavior from the original MAE environments instead of only cloning pretrained policies.

The implementation follows the paper direction at a local scale:

- PPO clipped objective
- GAE returns
- separate hider and seeker actor scopes
- role-specific value scopes
- centralized critic input built from each agent observation plus the full-agent observation context
- online observation normalization
- deterministic evaluation rollouts during training

The script is:

```text
scripts/train_mae_ppo.py
```

## Target Environments

Start with:

```text
multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet
```

Then scale to:

```text
multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet
```

Both environments are official examples from the OpenAI MAE repo. Both use 2 hiders and 2 seekers.

## Smoke-Tested Commands

Quadrant smoke:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_mae_ppo.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet \
    --out-dir /workspace/runs/mae_ppo_quadrant_smoke \
    --updates 1 \
    --rollout-steps 8 \
    --ppo-epochs 1 \
    --batch-size 16 \
    --hidden-sizes 64 \
    --eval-every 1 \
    --eval-episodes 1 \
    --save-every 1 \
    --seed 123
```

Validated output:

```text
saved_checkpoint: /workspace/runs/mae_ppo_quadrant_smoke/model.ckpt-1
saved_progress: /workspace/runs/mae_ppo_quadrant_smoke/progress.csv
saved_summary: /workspace/runs/mae_ppo_quadrant_smoke/summary.json
saved_report: /workspace/runs/mae_ppo_quadrant_smoke/report.txt
```

Episode-length quadrant smoke:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_mae_ppo.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet \
    --out-dir /workspace/runs/mae_ppo_quadrant_episode_smoke \
    --updates 1 \
    --rollout-steps 96 \
    --ppo-epochs 1 \
    --batch-size 64 \
    --hidden-sizes 64 \
    --eval-every 1 \
    --eval-episodes 1 \
    --save-every 1 \
    --seed 321
```

Validated episode accounting:

```text
update=1 episodes=1 hider_wins=0 seeker_wins=1 hider_return=-20.000 seeker_return=20.000
```

Full smoke:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_mae_ppo.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --out-dir /workspace/runs/mae_ppo_full_smoke \
    --updates 1 \
    --rollout-steps 8 \
    --ppo-epochs 1 \
    --batch-size 16 \
    --hidden-sizes 64 \
    --eval-every 1 \
    --eval-episodes 1 \
    --save-every 1 \
    --seed 123
```

Validated output:

```text
saved_checkpoint: /workspace/runs/mae_ppo_full_smoke/model.ckpt-1
saved_progress: /workspace/runs/mae_ppo_full_smoke/progress.csv
saved_summary: /workspace/runs/mae_ppo_full_smoke/summary.json
saved_report: /workspace/runs/mae_ppo_full_smoke/report.txt
```

## First Real Training Run

Use this as the first meaningful quadrant run:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_mae_ppo.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet \
    --out-dir /workspace/runs/mae_ppo_quadrant_v1 \
    --updates 300 \
    --rollout-steps 512 \
    --ppo-epochs 4 \
    --batch-size 512 \
    --hidden-sizes 256,256 \
    --learning-rate 0.0003 \
    --gamma 0.998 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --entropy-coef 0.01 \
    --eval-every 25 \
    --eval-episodes 5 \
    --save-every 25 \
    --seed 3001
```

After quadrant shows stable non-random behavior, start full:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_mae_ppo.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --out-dir /workspace/runs/mae_ppo_full_v1 \
    --updates 500 \
    --rollout-steps 512 \
    --ppo-epochs 4 \
    --batch-size 512 \
    --hidden-sizes 256,256 \
    --learning-rate 0.0003 \
    --gamma 0.998 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --entropy-coef 0.01 \
    --eval-every 25 \
    --eval-episodes 5 \
    --save-every 25 \
    --seed 4001
```

## Artifacts

Each run writes:

```text
model.ckpt-*
checkpoint
normalization.npz
progress.csv
summary.json
report.txt
hider_variables.txt
seeker_variables.txt
```

The hider and seeker models live in separate TensorFlow variable scopes:

```text
hider/*
seeker/*
```

`hider_variables.txt` and `seeker_variables.txt` list the exact role-specific variables saved in the checkpoint.

## What To Watch

Use `progress.csv` and `report.txt`.

Important columns:

- `recent_hider_wins`
- `recent_seeker_wins`
- `recent_mean_hider_return`
- `recent_mean_seeker_return`
- `approx_kl`
- `entropy`
- `clip_fraction`
- `recent_mean_episode_length`

Training should not be judged by one short smoke run. The smoke runs only prove that PPO collection, optimization, checkpointing, and evaluation execute correctly on both environments.

## Current Limitation

This is a local-scale PPO implementation, not a full reproduction of the OpenAI distributed system. It does not yet include:

- recurrent LSTM memory
- attention-based entity encoder
- checkpoint league/self-play population
- clone warm-start

Those are the next upgrades after the basic PPO-GAE loop produces stable learning curves on quadrant.
