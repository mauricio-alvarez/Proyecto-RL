# Random Rollout Collection

Validation date: 2026-06-08

## Command

```powershell
docker run --rm -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python /workspace/scripts/collect_rollouts.py --episodes 10 --steps 120 --seed 7 --out /workspace/runs/random_rollouts_hide_seek_quadrant.npz
```

## Output Artifact

```text
runs/random_rollouts_hide_seek_quadrant.npz
```

The file stores:
- `manifest`: environment path, seed, action space, observation space, and observation shape summary.
- `summaries`: per-episode step counts, rewards, done flags, discard flags, and final info keys.
- `episodes`: observations, actions, rewards, done flags, and info dictionaries for each step.

## Observed Result

Collected 10 random-policy episodes from `hide_and_seek_quadrant.jsonnet`.

All episodes reached 80 steps and ended normally:

```text
episode=0 steps=80 total_reward=[-36.0, -36.0, 36.0, 36.0] done=True discard=False
episode=1 steps=80 total_reward=[-26.0, -26.0, 26.0, 26.0] done=True discard=False
episode=2 steps=80 total_reward=[-2.0, -2.0, 2.0, 2.0] done=True discard=False
episode=3 steps=80 total_reward=[6.0, 6.0, -6.0, -6.0] done=True discard=False
episode=4 steps=80 total_reward=[4.0, 4.0, -4.0, -4.0] done=True discard=False
episode=5 steps=80 total_reward=[-20.0, -20.0, 20.0, 20.0] done=True discard=False
episode=6 steps=80 total_reward=[4.0, 4.0, -4.0, -4.0] done=True discard=False
episode=7 steps=80 total_reward=[-8.0, -8.0, 8.0, 8.0] done=True discard=False
episode=8 steps=80 total_reward=[16.0, 16.0, -16.0, -16.0] done=True discard=False
episode=9 steps=80 total_reward=[-4.0, -4.0, 4.0, 4.0] done=True discard=False
```

Dataset verification:

```text
npz_keys ['episodes', 'manifest', 'summaries']
episodes 10
summaries 10
first_summary {'episode': 0, 'steps': 80, 'total_reward': [-36.0, -36.0, 36.0, 36.0], 'done': True, 'discard_episode': False, 'last_info_keys': ['discard_episode', 'diverged', 'in_prep_phase', 'max_box_move', 'max_box_move_prep', 'max_ramp_move', 'max_ramp_move_prep', 'num_box_lock', 'num_box_lock_prep']}
```

