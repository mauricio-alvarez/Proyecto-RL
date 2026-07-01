# Proyecto RL: Linux Reproduction Guide

This repository contains a reproducible Docker setup for the legacy OpenAI multi-agent emergence environments. It is intended as the baseline for a seeker/hider reinforcement learning project.

The project now has two phases:

- MAE baseline reproduction and policy analysis.
- Principle-first cyber hide-and-seek RL environment and benchmark.
- Real-data cyber benchmark using CISA Known Exploited Vulnerabilities records.
- Local GPU-trained grid hide-and-seek models transferred into the KEV cyber benchmark.

The validated baseline is:

- Environment: `hide_and_seek_quadrant.jsonnet`
- Runtime: Docker image `mae-legacy:dev`
- Python: 3.6 inside Conda
- MuJoCo: 1.50
- Main validation: headless reset, stepping, and random rollout collection

## What Is Already Implemented

Implemented:

- Legacy Docker image definition in `docker/Dockerfile.mae-legacy`.
- Conda dependency file in `environment.yml`.
- Headless smoke test in `scripts/smoke_mae.py`.
- Rollout collector in `scripts/collect_rollouts.py`.
- Pretrained policy rollout collector in `scripts/collect_policy_rollouts.py`.
- Single-policy evaluator in `scripts/evaluate_policy.py`.
- Cross-play hider/seeker evaluator in `scripts/evaluate_crossplay.py`.
- Batch policy benchmark scripts in `scripts/benchmark_policies.py` and `scripts/benchmark_crossplay.py`.
- PPO + GAE MAE self-play trainer in `scripts/train_mae_ppo.py`.
- MAE PPO to KEV zero-shot transfer evaluator in `scripts/evaluate_mae_ppo_kev_transfer.py`.
- Behavioral dataset collector in `scripts/collect_behavioral_dataset.py`.
- Behavioral dataset inspector in `scripts/inspect_behavioral_dataset.py`.
- Behavior-cloning baseline trainer in `scripts/train_behavior_clone.py`.
- Structured behavior-cloning trainer in `scripts/train_structured_behavior_clone.py`.
- DAgger dataset collector in `scripts/collect_dagger_dataset.py`.
- Behavioral dataset merger in `scripts/merge_behavioral_datasets.py`.
- Closed-loop cloned-vs-pretrained evaluator in `scripts/evaluate_clone_vs_pretrained.py`, including separate hider and seeker clone checkpoints.
- Principle-first cyber RL environment in `cyber_rl/`.
- Cyber RL smoke test in `scripts/smoke_cyber_rl.py`.
- Cyber RL benchmark runner in `scripts/run_cyber_benchmarks.py`.
- CISA KEV downloader in `scripts/fetch_cisa_kev.py`.
- CISA KEV real-data benchmark runner in `scripts/run_kev_benchmarks.py`.
- Grid-to-KEV model transfer trainer in `scripts/train_grid_to_kev_transfer.py`.
- OpenGL diagnostic script in `scripts/check_gl.py`.
- Methodology and validation notes in `docs/`.
- A sample random rollout artifact in `runs/random_rollouts_hide_seek_quadrant.npz`.

Not implemented:

- Full OpenAI-scale distributed MAE training.
- Real cyber-range integration.
- Production-quality trajectory visualization.

## Repository Layout

```text
.
|-- docker/
|   `-- Dockerfile.mae-legacy
|-- docs/
|   |-- cyber_principle_first.md
|   |-- environment_validation.md
|   |-- grid_to_kev_model_transfer.md
|   |-- kev_realworld_pipeline.md
|   |-- mae_separated_models.md
|   |-- mae_ppo_training.md
|   |-- mae_ppo_to_kev_transfer.md
|   |-- methodology.md
|   `-- rollout_collection.md
|-- cyber_rl/
|   |-- benchmark.py
|   |-- env.py
|   |-- policies.py
|   `-- q_learning.py
|-- mujoco-worldgen/
|-- multi-agent-emergence-environments/
|-- runs/
|   `-- random_rollouts_hide_seek_quadrant.npz
|-- scripts/
|   |-- benchmark_crossplay.py
|   |-- benchmark_policies.py
|   |-- check_gl.py
|   |-- collect_behavioral_dataset.py
|   |-- collect_dagger_dataset.py
|   |-- collect_policy_rollouts.py
|   |-- collect_rollouts.py
|   |-- evaluate_crossplay.py
|   |-- evaluate_clone_vs_pretrained.py
|   |-- evaluate_policy.py
|   |-- fetch_cisa_kev.py
|   |-- inspect_behavioral_dataset.py
|   |-- merge_behavioral_datasets.py
|   |-- render_rollout_video.py
|   |-- run_kev_benchmarks.py
|   |-- run_cyber_benchmarks.py
|   |-- smoke_cyber_rl.py
|   |-- smoke_mae.py
|   |-- train_behavior_clone.py
|   |-- train_grid_to_kev_transfer.py
|   |-- train_mae_ppo.py
|   |-- evaluate_mae_ppo_kev_transfer.py
|   |-- train_structured_behavior_clone.py
|   `-- verify_policy_deps.py
|-- ASIS.md
|-- environment.yml
`-- README.md
```

## Run The Cyber Principle Benchmark

The cyber benchmark is independent of MuJoCo rendering. It reuses the same Docker image only as a known Python runtime.

Smoke test:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/smoke_cyber_rl.py
```

Validated output:

```text
cyber_smoke_ok
benchmark_rows: 80
pair_summaries: 20
```

Full benchmark:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/run_cyber_benchmarks.py \
    --out-dir /workspace/runs/cyber_principle_benchmark_v4 \
    --episodes-per-family 50 \
    --q-train-episodes 8000 \
    --linear-q-train-episodes 12000 \
    --seed 1500 \
    --n-nodes 8 \
    --max-steps 24
```

Validated benchmark evidence:

```text
episodes: 7500
pair_summaries: 30
attacker=targeted success=0.399 caught=0.601 return=-12.47
attacker=linear_q success=0.190 caught=0.014 return=-62.23
attacker=stealth success=0.075 caught=0.497 return=-77.48
attacker=greedy success=0.054 caught=0.946 return=-81.95
attacker=random success=0.035 caught=0.799 return=-89.33
attacker=q_table success=0.017 caught=0.222 return=-92.23
```

Main outputs:

```text
runs/cyber_principle_benchmark_v4/report.md
runs/cyber_principle_benchmark_v4/summary.json
runs/cyber_principle_benchmark_v4/episodes.csv
runs/cyber_principle_benchmark_v4/pair_summaries.csv
runs/cyber_principle_benchmark_v4/attacker_success.svg
```

Design notes and interpretation are in `docs/cyber_principle_first.md`.

## Run The Real-Data KEV Benchmark

The real-data benchmark uses CISA's Known Exploited Vulnerabilities catalog. It maps real CVE/vendor/product/CWE metadata into safe abstract cyber scenarios. It does not use exploit code or live targets.

Fetch the dataset:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/fetch_cisa_kev.py \
    --out /workspace/data/raw/known_exploited_vulnerabilities.json
```

Validated dataset metadata:

```text
title: CISA Catalog of Known Exploited Vulnerabilities
catalogVersion: 2026.06.25
dateReleased: 2026-06-25T19:03:21.8037Z
count: 1629
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

Validated real-data benchmark evidence:

```text
episodes: 7500
pair_summaries: 30
attacker=targeted success=0.281 caught=0.719 timeout=0.000 return=-34.02
attacker=linear_q_kev success=0.126 caught=0.006 timeout=0.868 return=-75.85
attacker=stealth success=0.047 caught=0.442 timeout=0.510 return=-81.14
attacker=greedy success=0.027 caught=0.973 timeout=0.000 return=-88.06
attacker=random success=0.018 caught=0.795 timeout=0.187 return=-91.90
attacker=q_table_kev success=0.007 caught=0.198 timeout=0.795 return=-95.27
```

Main outputs:

```text
runs/kev_realworld_benchmark_v1/report.md
runs/kev_realworld_benchmark_v1/summary.json
runs/kev_realworld_benchmark_v1/episodes.csv
runs/kev_realworld_benchmark_v1/pair_summaries.csv
runs/kev_realworld_benchmark_v1/attacker_success.svg
runs/kev_realworld_benchmark_v1/kev_scenario_samples.json
```

Full design and interpretation are in `docs/kev_realworld_pipeline.md`.

## Train Game Models And Transfer To KEV

This path trains two local game models first:

- `grid_seeker.pt`
- `grid_hider.pt`

Then it transfers their encoder weights into cyber models:

- `transfer_attacker.pt`
- `transfer_defender.pt`

This run uses GPU through the PyTorch CUDA image. The legacy MAE Docker image is not used for this training step because it is CPU-only and does not include PyTorch.

Run:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime \
  python /workspace/scripts/train_grid_to_kev_transfer.py \
    --kev-json /workspace/data/raw/known_exploited_vulnerabilities.json \
    --out-dir /workspace/runs/grid_to_kev_transfer_v1 \
    --grid-samples 80000 \
    --cyber-samples 50000 \
    --grid-epochs 25 \
    --cyber-epochs 25 \
    --episodes-per-family 35 \
    --seed 3200 \
    --n-nodes 8 \
    --max-steps 24
```

Validated GPU:

```text
NVIDIA GeForce RTX 3060 Ti
PyTorch 2.4.1+cu124
CUDA available: True
```

Validated model results:

```text
grid_seeker validation accuracy: 0.998
grid_hider validation accuracy: 0.989

cyber attacker imitation:
transfer: 0.7354
scratch:  0.7409

cyber defender imitation:
transfer: 0.7569
scratch:  0.7613

closed-loop KEV attacker success:
targeted:          0.161
transfer_attacker: 0.103
scratch_attacker:  0.101

attacker success against defenders:
adaptive:          0.095
decoy_frontier:    0.097
transfer_defender: 0.137
scratch_defender:  0.158
```

Main outputs:

```text
runs/grid_to_kev_transfer_v1/grid_seeker.pt
runs/grid_to_kev_transfer_v1/grid_hider.pt
runs/grid_to_kev_transfer_v1/transfer_attacker.pt
runs/grid_to_kev_transfer_v1/transfer_defender.pt
runs/grid_to_kev_transfer_v1/scratch_attacker.pt
runs/grid_to_kev_transfer_v1/scratch_defender.pt
runs/grid_to_kev_transfer_v1/transfer_report.md
runs/grid_to_kev_transfer_v1/transfer_summary.json
```

Full interpretation is in `docs/grid_to_kev_model_transfer.md`.

## Linux Prerequisites

Install these on the Linux host:

- `git`
- Docker Engine or Docker Desktop for Linux
- Optional for live GUI visualization: working X11 or XWayland session

On Ubuntu/Debian, a minimal Docker setup is:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Optional: allow your user to run Docker without `sudo`.

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Verify Docker:

```bash
docker version
docker run --rm hello-world
```

If your environment requires `sudo docker`, use `sudo docker` in the commands below.

## Prepare The Workspace

Clone this project repository:

```bash
git clone <PROJECT_REPO_URL> proyecto-rl
cd proyecto-rl
```

If the two upstream source directories are not already present, clone them:

```bash
git clone https://github.com/openai/mujoco-worldgen.git
git clone https://github.com/openai/multi-agent-emergence-environments.git
```

The Docker build expects both directories to exist at the repository root:

```text
./mujoco-worldgen
./multi-agent-emergence-environments
```

## Build The Docker Image

From the repository root:

```bash
docker build -f docker/Dockerfile.mae-legacy -t mae-legacy:dev .
```

Expected final build evidence:

```text
naming to docker.io/library/mae-legacy:dev
```

The image build installs:

- Python 3.6 through Conda
- MuJoCo 1.50 at `/root/.mujoco/mjpro150`
- Public legacy MuJoCo key at `/root/.mujoco/mjkey.txt`
- `mujoco-py>=1.50.1,<1.50.2`
- `gym==0.10.8`
- `scipy==1.3.1`
- `jsonnet==0.17.0`
- `tensorflow==1.13.1`
- `cloudpickle==0.5.2`
- `opencv-python==4.5.5.64`
- `baselines==0.1.5` installed with `--no-deps` to avoid unnecessary MPI training dependencies
- Editable installs of `mujoco-worldgen` and `multi-agent-emergence-environments`

## Run A Headless Smoke Test

This verifies that the container can import MuJoCo, load the JSONNet environment, reset it, and step it.

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/smoke_mae.py --episodes 3 --steps 20
```

Expected output shape:

```text
action_space: Dict(...)
observation_keys: [...]
episode=0 steps=20 total_reward=[...] done=False discard=False
episode=1 steps=20 total_reward=[...] done=False discard=False
episode=2 steps=20 total_reward=[...] done=False discard=False
```

`discard=False` means MuJoCo did not throw a simulation error.

## Collect Random Rollouts

This saves observations, actions, rewards, done flags, and info dictionaries into a compressed `.npz`.

```bash
mkdir -p runs
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_rollouts.py \
    --episodes 10 \
    --steps 120 \
    --seed 7 \
    --out /workspace/runs/random_rollouts_hide_seek_quadrant.npz
```

Validated result from the current workspace:

```text
episodes_collected: 10
discard_count: 0
saved: /workspace/runs/random_rollouts_hide_seek_quadrant.npz
```

The output file on the Linux host is:

```text
./runs/random_rollouts_hide_seek_quadrant.npz
```

## Run Pretrained Hide-And-Seek Policies

The upstream repository includes pretrained `.npz` policy weights. The currently validated hide-and-seek pair is:

```text
examples/hide_and_seek_quadrant.jsonnet
examples/hide_and_seek_quadrant.npz
```

Verify policy playback dependencies:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/verify_policy_deps.py
```

Expected output:

```text
tensorflow: 1.13.1
opencv: 4.5.5
baselines.make_pdtype: True
load_policy: True
policy_deps_ok: True
```

Collect pretrained policy rollouts:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_policy_rollouts.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet \
    --policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.npz \
    --episodes 3 \
    --steps 120 \
    --seed 21 \
    --out /workspace/runs/policy_rollouts_hide_seek_quadrant.npz
```

Validated result from the current workspace:

```text
episode=0 seed=21 steps=80 total_reward=[48.0, 48.0, -48.0, -48.0] done=True discard=False
episode=1 seed=22 steps=80 total_reward=[48.0, 48.0, -48.0, -48.0] done=True discard=False
episode=2 seed=23 steps=80 total_reward=[48.0, 48.0, -48.0, -48.0] done=True discard=False
saved: /workspace/runs/policy_rollouts_hide_seek_quadrant.npz
episodes_collected: 3
discard_count: 0
```

Render the pretrained rollout:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/render_rollout_video.py \
    --input /workspace/runs/policy_rollouts_hide_seek_quadrant.npz \
    --episode 0 \
    --out /workspace/videos/policy_rollout_hide_seek_quadrant_ep0.gif \
    --fps 12
```

Expected render output includes wall confirmation:

```text
walls_drawn: True
```

Validated output artifact:

```text
./videos/policy_rollout_hide_seek_quadrant_ep0.gif
```

Other hide-and-seek weights available in the upstream examples:

```text
examples/hide_and_seek_full.npz
examples/hide_and_seek_quadrant_physics_exploits.npz
examples/hide_and_seek_policy_phases/a_chasing.npz
examples/hide_and_seek_policy_phases/b_forts.npz
examples/hide_and_seek_policy_phases/c_ramps.npz
examples/hide_and_seek_policy_phases/d_ramp_defense.npz
examples/hide_and_seek_policy_phases/e_box_surfing.npz
```

Evaluate an existing policy over fixed seeds:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_policy.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet \
    --policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.npz \
    --episodes 20 \
    --steps 120 \
    --seed 21 \
    --out /workspace/runs/eval_hide_and_seek_quadrant.json
```

Try the existing phase checkpoints by changing `--policy`, for example:

```text
/workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/a_chasing.npz
/workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/b_forts.npz
/workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz
/workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz
/workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/e_box_surfing.npz
```

The evaluator reports `hider_wins`, `seeker_wins`, mean team returns, and the fraction of steps where any seeker can see a hider.

Some checkpoints may fail to load in a specific environment with a shape error such as:

```text
Error assigning weights of shape (15,) to ... box_obs ... shape=(12,)
```

That means the checkpoint was trained with a different observation schema than the selected `.jsonnet` environment. Use the batch video sweep below to try every existing policy and keep going when a checkpoint is incompatible.

Create one rollout video per compatible policy:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/sweep_policy_videos.py \
    --env auto \
    --episodes 1 \
    --steps 260 \
    --seed 21 \
    --out-dir /workspace/runs/policy_sweep \
    --video-dir /workspace/videos/policy_sweep \
    --summary-out /workspace/runs/policy_sweep/summary.json
```

On Windows PowerShell, use the same command with the local absolute volume:

```powershell
docker run --rm `
  -v "C:\Users\asus\Documents\Proyecto RL:/workspace" `
  mae-legacy:dev `
  conda run --no-capture-output -n mae-legacy `
  python /workspace/scripts/sweep_policy_videos.py `
    --env auto `
    --episodes 1 `
    --steps 260 `
    --seed 21 `
    --out-dir /workspace/runs/policy_sweep `
    --video-dir /workspace/videos/policy_sweep `
    --summary-out /workspace/runs/policy_sweep/summary.json
```

Generated videos are saved under:

```text
videos/policy_sweep/
```

The compatibility summary is saved to:

```text
runs/policy_sweep/summary.json
```

With `--env auto`, the sweep uses the quadrant environment for `hide_and_seek_quadrant*.npz` and the full hide-and-seek environment for `hide_and_seek_full.npz` plus the phase checkpoints.

## Benchmark Existing Policies

Use the benchmark script to evaluate every existing hide-and-seek policy over multiple seeds, write a plain-text report, and create one plot per policy.

Linux:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/benchmark_policies.py \
    --env auto \
    --episodes 20 \
    --steps 260 \
    --seed 21 \
    --out-dir /workspace/runs/policy_benchmark \
    --plot-dir /workspace/plots/policy_benchmark \
    --report-out /workspace/runs/policy_benchmark/benchmark_report.txt \
    --summary-out /workspace/runs/policy_benchmark/benchmark_summary.json
```

Windows PowerShell:

```powershell
docker run --rm `
  -v "C:\Users\asus\Documents\Proyecto RL:/workspace" `
  mae-legacy:dev `
  conda run --no-capture-output -n mae-legacy `
  python /workspace/scripts/benchmark_policies.py `
    --env auto `
    --episodes 20 `
    --steps 260 `
    --seed 21 `
    --out-dir /workspace/runs/policy_benchmark `
    --plot-dir /workspace/plots/policy_benchmark `
    --report-out /workspace/runs/policy_benchmark/benchmark_report.txt `
    --summary-out /workspace/runs/policy_benchmark/benchmark_summary.json
```

Outputs:

```text
runs/policy_benchmark/benchmark_report.txt
runs/policy_benchmark/benchmark_summary.json
plots/policy_benchmark/aggregate_summary.png
plots/policy_benchmark/<policy_name>.png
```

The report ranks policies by seeker win rate and mean visible fraction. It includes seeker win rate, hider win rate, and tie rate explicitly. The aggregate plot shows seeker win rate, hider win rate, tie rate, and visible fraction for each policy. Each policy plot shows mean returns, visible fraction, and first visible step across evaluation seeds.

## Cross-Play Hider Vs Seeker Policies

The single-policy benchmark uses one checkpoint for all agents. Cross-play is stricter: it uses one checkpoint for the hider agents and a different checkpoint for the seeker agents in the same environment. This is the current best tool for choosing baseline hider and seeker controllers before adding new training.

Run one matchup:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_crossplay.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --hider-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz \
    --seeker-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/a_chasing.npz \
    --episodes 10 \
    --steps 260 \
    --seed 21 \
    --out /workspace/runs/crossplay_smoke/d_ramp_defense_vs_a_chasing.json
```

Run the validated selected matrix:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/benchmark_crossplay.py \
    --env full \
    --hider-policy hide_and_seek_policy_phases/d_ramp_defense.npz \
    --hider-policy hide_and_seek_policy_phases/e_box_surfing.npz \
    --hider-policy hide_and_seek_policy_phases/b_forts.npz \
    --seeker-policy hide_and_seek_policy_phases/a_chasing.npz \
    --seeker-policy hide_and_seek_full.npz \
    --seeker-policy hide_and_seek_policy_phases/c_ramps.npz \
    --episodes 10 \
    --steps 260 \
    --seed 21 \
    --out-dir /workspace/runs/crossplay_benchmark_full_selected_10 \
    --plot-dir /workspace/plots/crossplay_benchmark_full_selected_10 \
    --report-out /workspace/runs/crossplay_benchmark_full_selected_10/crossplay_report.txt \
    --summary-out /workspace/runs/crossplay_benchmark_full_selected_10/crossplay_summary.json
```

Outputs:

```text
runs/crossplay_benchmark_full_selected_10/crossplay_report.txt
runs/crossplay_benchmark_full_selected_10/crossplay_summary.json
plots/crossplay_benchmark_full_selected_10/crossplay_seeker_win_rate.png
plots/crossplay_benchmark_full_selected_10/crossplay_hider_win_rate.png
plots/crossplay_benchmark_full_selected_10/crossplay_visible_fraction.png
```

Validated selected-matrix evidence from the current workspace:

```text
matchup_count: 9
failed_count: 0
best seeker matchup: hider=phase_b_forts, seeker=phase_c_ramps, seeker_win_rate=1.000
best hider matchup: hider=phase_d_ramp_defense, seeker=phase_a_chasing, hider_win_rate=0.900
```

To regenerate only the report and plots from already completed matchup JSON files, add:

```text
--reuse-existing
```

Run the full 6x6 full-environment matrix by omitting the repeated `--hider-policy` and `--seeker-policy` arguments:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/benchmark_crossplay.py \
    --env full \
    --episodes 10 \
    --steps 260 \
    --seed 21 \
    --out-dir /workspace/runs/crossplay_benchmark_full_10 \
    --plot-dir /workspace/plots/crossplay_benchmark_full_10 \
    --report-out /workspace/runs/crossplay_benchmark_full_10/crossplay_report.txt \
    --summary-out /workspace/runs/crossplay_benchmark_full_10/crossplay_summary.json
```

Interpretation:

- Rows are hider policies.
- Columns are seeker policies.
- Seeker win rate close to `1.0` means the seeker controller reliably finds/catches the hiders in that matchup.
- Hider win rate close to `1.0` means the hider controller reliably survives or avoids being caught.
- Visible fraction is the fraction of rollout steps where at least one seeker can see at least one hider. It is not itself a win condition, but it helps diagnose whether a seeker is exploring/searching effectively.

## Train Hider And Seeker With PPO + GAE

This is now the main from-environment training path. It trains separate hider and seeker actor scopes with PPO clipping and GAE returns.

Main script:

```text
scripts/train_mae_ppo.py
```

Target environments:

```text
multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet
multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet
```

Smoke-tested quadrant command:

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

Smoke-tested full command:

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

Validated outputs for both smoke runs:

```text
model.ckpt-1
normalization.npz
progress.csv
summary.json
report.txt
hider_variables.txt
seeker_variables.txt
```

The first real run should be quadrant, then full:

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

Full-environment command:

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

Full details are in `docs/mae_ppo_training.md`.

## Build A Behavioral Dataset

The behavioral dataset converts policy rollouts into supervised learning rows. Each row is one `(episode, step, agent)` sample with:

- `obs_*`: pre-action observation tensors for one agent.
- `action_*`: action labels from the policy.
- `next_obs_*`: post-step observation tensors for the same agent.
- `role`: `0` for hider, `1` for seeker.
- `policy_id`: `0` for the hider policy, `1` for the seeker policy.
- `reward`, `done`, `visible_any`, `episode_winner`, `episode`, `step`, and `agent_index`.

By default, `collect_behavioral_dataset.py` uses deterministic policy actions, meaning it records the mode of each pretrained policy distribution. This is the preferred target for behavior cloning. Add `--stochastic-policy` only when you explicitly want sampled policy actions.

Generate the validated sample dataset:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_behavioral_dataset.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --hider-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz \
    --seeker-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz \
    --episodes 3 \
    --steps 260 \
    --seed 101 \
    --out /workspace/runs/behavioral_hide_seek_full_d_ramp_vs_c_ramps_det_3ep.npz
```

Inspect the dataset:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/inspect_behavioral_dataset.py \
    /workspace/runs/behavioral_hide_seek_full_d_ramp_vs_c_ramps_det_3ep.npz
```

Validated evidence from the current workspace:

```text
schema_version: behavioral_dataset_v1
episodes_collected: 3
samples: 2880
discard_count: 0
role_counts: {'0': 1440, '1': 1440}
winner_counts: {'0': 1920, '1': 960}
obs_agent_qpos_qvel: [2880, 3, 10]
obs_box_obs: [2880, 9, 15]
obs_lidar: [2880, 30, 1]
action_action_movement: [2880, 3]
action_action_pull: [2880]
action_action_glueall: [2880]
```

For a larger first training dataset, increase `--episodes`:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_behavioral_dataset.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --hider-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz \
    --seeker-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz \
    --episodes 20 \
    --steps 260 \
    --seed 1001 \
    --out /workspace/runs/behavioral_hide_seek_full_d_ramp_vs_c_ramps_det_20ep.npz
```

Validated 20-episode dataset evidence from the current workspace:

```text
episodes_collected: 20
samples: 19200
discard_count: 0
role_counts: {'0': 9600, '1': 9600}
winner_counts: {'0': 18240, '1': 960}
obs_agent_qpos_qvel: [19200, 3, 10]
obs_box_obs: [19200, 9, 15]
obs_lidar: [19200, 30, 1]
action_action_movement: [19200, 3]
```

This dataset is suitable for behavior cloning, role-conditioned policy modeling, visibility prediction, and representation learning before moving to a modern training stack.

## Train A Behavior-Cloning Baseline

The baseline trainer loads a behavioral `.npz`, splits by episode, concatenates selected `obs_*` tensors plus the role label, and trains a TensorFlow 1.x MLP with five classification heads:

- `movement_0`, `movement_1`, `movement_2`: 11-class movement action dimensions.
- `pull`: binary action.
- `glueall`: binary action.

Train the validated baseline:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_behavior_clone.py \
    --dataset /workspace/runs/behavioral_hide_seek_full_d_ramp_vs_c_ramps_det_20ep.npz \
    --out-dir /workspace/runs/bc_baseline_d_ramp_vs_c_ramps_det_20ep \
    --epochs 20 \
    --batch-size 512 \
    --hidden-sizes 256,128 \
    --learning-rate 0.001 \
    --seed 123
```

Outputs:

```text
runs/bc_baseline_d_ramp_vs_c_ramps_det_20ep/model.ckpt.*
runs/bc_baseline_d_ramp_vs_c_ramps_det_20ep/preprocessing.npz
runs/bc_baseline_d_ramp_vs_c_ramps_det_20ep/metrics.json
runs/bc_baseline_d_ramp_vs_c_ramps_det_20ep/metrics.txt
```

Validated baseline evidence:

```text
train_samples: 15360
val_samples: 3840
final validation movement_exact: 0.110937
majority validation movement_exact: 0.017188
final validation action_exact: 0.051562
majority validation action_exact: 0.001042
final validation movement accuracies: 0.338021, 0.483594, 0.403646
final validation pull accuracy: 0.736458
final validation glueall accuracy: 0.684635
```

`action_exact` is strict: all three movement dimensions plus `pull` and `glueall` must match simultaneously. For diagnosing early models, use per-head accuracy and `movement_exact` before optimizing for full-action exact match.

## Train A Structured Behavior Clone

The structured clone is the stronger implementation. It replaces the flat observation MLP with:

- Separate encoders for self observation and lidar.
- Mask-aware entity encoders for agents, boxes, and ramps.
- Role-specific action heads for hiders and seekers.
- Optional class-balanced action losses.

Train the validated 100-episode structured clone:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_structured_behavior_clone.py \
    --dataset /workspace/runs/behavioral_hide_seek_full_d_ramp_vs_c_ramps_det_100ep.npz \
    --out-dir /workspace/runs/bc_structured_d_ramp_vs_c_ramps_det_100ep \
    --epochs 40 \
    --batch-size 1024 \
    --hidden-sizes 256,256 \
    --entity-hidden 64 \
    --learning-rate 0.001 \
    --seed 123
```

Validated structured-clone offline evidence:

```text
dataset samples: 96000
train samples: 76800
validation samples: 19200
validation movement_exact: 0.149115
majority movement_exact:   0.002760
validation action_exact:   0.123646
majority action_exact:     0.001146
validation movement acc:   0.377917, 0.518385, 0.551302
validation pull acc:       0.833542
validation glue acc:       0.843594
```

Compared with the flat 20-episode baseline, the structured 100-episode clone improves offline imitation substantially. It still does not solve closed-loop policy robustness by itself.

## Evaluate Clone Vs Pretrained Policies

Offline behavior-cloning accuracy is not enough. The cloned model must also be tested in closed-loop interaction because prediction errors change the next observation and can quickly move the policy into states not represented in the dataset.

Run the validated deterministic comparison:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_clone_vs_pretrained.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --hider-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz \
    --seeker-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz \
    --clone-dir /workspace/runs/bc_baseline_d_ramp_vs_c_ramps_det_20ep \
    --episodes 10 \
    --steps 260 \
    --seed 2001 \
    --out /workspace/runs/clone_vs_pretrained_det_10/summary.json \
    --report-out /workspace/runs/clone_vs_pretrained_det_10/report.txt
```

Outputs:

```text
runs/clone_vs_pretrained_det_10/summary.json
runs/clone_vs_pretrained_det_10/report.txt
```

Validated closed-loop evidence:

```text
case                            S    H    T    S_win    H_win   visible   S_return   H_return
pretrained_vs_pretrained        1    9    0    0.100    0.900    0.309   -120.400    120.400
clone_hider_vs_pretrained      10    0    0    1.000    0.000    0.762    110.950   -161.400
pretrained_vs_clone_seeker      0   10    0    0.000    1.000    0.357   -297.950    139.600
clone_vs_clone                  4    6    0    0.400    0.600    0.507   -107.400    -55.050
```

Interpretation:

- The pretrained-vs-pretrained control reproduces strong hider behavior on these seeds.
- The cloned hider collapses against the pretrained seeker.
- The cloned seeker fails against the pretrained hider.
- Clone-vs-clone has mixed winners, but both sides have poor returns, so this should not be interpreted as robust cloned behavior.

The conclusion is that the current clone is a useful baseline, but not a deployable behavioral policy. The next improvement should be a larger and more structured clone: more episodes, multiple policy matchups, entity-aware encoders, role-specific diagnostics, and likely DAgger-style dataset aggregation.

Run the same closed-loop comparison with the structured 100-episode clone:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_clone_vs_pretrained.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --hider-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz \
    --seeker-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz \
    --clone-dir /workspace/runs/bc_structured_d_ramp_vs_c_ramps_det_100ep \
    --episodes 10 \
    --steps 260 \
    --seed 2001 \
    --out /workspace/runs/clone_vs_pretrained_structured_det_100ep_10/summary.json \
    --report-out /workspace/runs/clone_vs_pretrained_structured_det_100ep_10/report.txt
```

Validated structured closed-loop evidence:

```text
case                            S    H    T    S_win    H_win   visible   S_return   H_return
pretrained_vs_pretrained        0   10    0    0.000    1.000    0.265   -143.800    143.800
clone_hider_vs_pretrained      10    0    0    1.000    0.000    0.764    129.200   -149.450
pretrained_vs_clone_seeker      0   10    0    0.000    1.000    0.298   -150.300    144.000
clone_vs_clone                  5    4    1    0.500    0.400    0.515     14.850    -16.950
```

This means architecture plus 100 expert episodes improves imitation metrics, but still fails the policy-equivalence requirement. The next required method is dataset aggregation: run the clone, collect the states it actually visits, label those states with the pretrained expert, retrain, and repeat.

## DAgger Iteration 1

DAgger is implemented as an additional dataset collection phase:

- The behavior policy runs in the environment and creates the visited-state distribution.
- The deterministic pretrained policies label those visited states.
- The expert labels are stored in the normal `action_*` training keys.
- The executed behavior actions are stored separately as `behavior_action_*` for diagnostics.

The current implementation is:

- `scripts/collect_dagger_dataset.py`
- `scripts/merge_behavioral_datasets.py`

Collect hider-side correction data, where the cloned hiders act against the expert seeker:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_dagger_dataset.py \
    --clone-dir /workspace/runs/bc_structured_d_ramp_vs_c_ramps_det_100ep \
    --execute-hider clone \
    --execute-seeker expert \
    --episodes 20 \
    --steps 260 \
    --seed 4001 \
    --out /workspace/runs/dagger_hider_clone_vs_expert_20ep.npz
```

Collect seeker-side correction data, where the expert hiders act against the cloned seeker:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_dagger_dataset.py \
    --clone-dir /workspace/runs/bc_structured_d_ramp_vs_c_ramps_det_100ep \
    --execute-hider expert \
    --execute-seeker clone \
    --episodes 20 \
    --steps 260 \
    --seed 5001 \
    --out /workspace/runs/dagger_expert_vs_seeker_clone_20ep.npz
```

Collect deployment-distribution correction data, where both cloned roles act:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_dagger_dataset.py \
    --clone-dir /workspace/runs/bc_structured_d_ramp_vs_c_ramps_det_100ep \
    --execute-hider clone \
    --execute-seeker clone \
    --episodes 20 \
    --steps 260 \
    --seed 6001 \
    --out /workspace/runs/dagger_clone_vs_clone_20ep.npz
```

Merge the original expert dataset plus the three DAgger correction datasets:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/merge_behavioral_datasets.py \
    --out /workspace/runs/behavioral_dagger_iter1_merged.npz \
    /workspace/runs/behavioral_hide_seek_full_d_ramp_vs_c_ramps_det_100ep.npz \
    /workspace/runs/dagger_hider_clone_vs_expert_20ep.npz \
    /workspace/runs/dagger_expert_vs_seeker_clone_20ep.npz \
    /workspace/runs/dagger_clone_vs_clone_20ep.npz
```

Validated merge evidence:

```text
samples: 153600
episodes_collected: 160
inputs:
- expert behavioral dataset: 100 episodes, 96000 samples
- DAgger hider clone vs expert seeker: 20 episodes, 19200 samples
- DAgger expert hider vs seeker clone: 20 episodes, 19200 samples
- DAgger clone vs clone: 20 episodes, 19200 samples
```

Train the first aggregated clone:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_structured_behavior_clone.py \
    --dataset /workspace/runs/behavioral_dagger_iter1_merged.npz \
    --out-dir /workspace/runs/bc_structured_dagger_iter1 \
    --epochs 40 \
    --batch-size 1024 \
    --hidden-sizes 256,256 \
    --entity-hidden 64 \
    --learning-rate 0.001 \
    --seed 123
```

Validated offline result:

```text
samples: 153600
train_samples: 122880
val_samples: 30720
final_val action_exact: 0.097493
final_val movement_exact: 0.124056
final_val acc_movement_0: 0.373405
final_val acc_movement_1: 0.481152
final_val acc_movement_2: 0.483008
final_val acc_pull: 0.802767
final_val acc_glueall: 0.787858
```

Run the same closed-loop evaluation:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_clone_vs_pretrained.py \
    --clone-dir /workspace/runs/bc_structured_dagger_iter1 \
    --episodes 10 \
    --steps 260 \
    --seed 2001 \
    --out /workspace/runs/clone_vs_pretrained_dagger_iter1_10/summary.json \
    --report-out /workspace/runs/clone_vs_pretrained_dagger_iter1_10/report.txt
```

Validated closed-loop result:

```text
case                            S    H    T    S_win    H_win   T_rate  visible   S_return   H_return
pretrained_vs_pretrained        1    9    0    0.100    0.900    0.000    0.319   -120.400    120.400
clone_hider_vs_pretrained      10    0    0    1.000    0.000    0.000    0.668    127.600   -127.600
pretrained_vs_clone_seeker      0   10    0    0.000    1.000    0.000    0.398   -144.000    144.000
clone_vs_clone                  7    3    0    0.700    0.300    0.000    0.556     34.550    -49.600
```

Interpretation: DAgger iteration 1 is implemented and reproducible, but it is not yet enough to make the clone policy-equivalent to the pretrained policies. The next experimental step is not to move to cybersecurity yet; it is to run iterative DAgger rounds with larger correction datasets and likely split role-specific or recurrent clone models, then only transfer once closed-loop hider and seeker behavior are both stable against the pretrained counterpart.

## Extensive DAgger Iteration 2

The second DAgger round starts from `runs/bc_structured_dagger_iter1` and expands the correction data from 60 DAgger episodes to 220 DAgger episodes total.

Collect four iteration-2 correction datasets:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_dagger_dataset.py \
    --clone-dir /workspace/runs/bc_structured_dagger_iter1 \
    --execute-hider clone \
    --execute-seeker expert \
    --episodes 40 \
    --steps 260 \
    --seed 7001 \
    --out /workspace/runs/dagger_iter2_hider_clone_vs_expert_40ep.npz

docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_dagger_dataset.py \
    --clone-dir /workspace/runs/bc_structured_dagger_iter1 \
    --execute-hider expert \
    --execute-seeker clone \
    --episodes 40 \
    --steps 260 \
    --seed 8001 \
    --out /workspace/runs/dagger_iter2_expert_vs_seeker_clone_40ep.npz

docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_dagger_dataset.py \
    --clone-dir /workspace/runs/bc_structured_dagger_iter1 \
    --execute-hider clone \
    --execute-seeker clone \
    --episodes 40 \
    --steps 260 \
    --seed 9001 \
    --out /workspace/runs/dagger_iter2_clone_vs_clone_40ep.npz

docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/collect_dagger_dataset.py \
    --clone-dir /workspace/runs/bc_structured_dagger_iter1 \
    --execute-hider mix \
    --execute-seeker mix \
    --mix-expert-prob 0.2 \
    --episodes 40 \
    --steps 260 \
    --seed 10001 \
    --out /workspace/runs/dagger_iter2_mixed_20pct_expert_40ep.npz
```

Merge all expert, iteration-1, and iteration-2 datasets:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/merge_behavioral_datasets.py \
    --out /workspace/runs/behavioral_dagger_iter2_extensive_merged.npz \
    /workspace/runs/behavioral_hide_seek_full_d_ramp_vs_c_ramps_det_100ep.npz \
    /workspace/runs/dagger_hider_clone_vs_expert_20ep.npz \
    /workspace/runs/dagger_expert_vs_seeker_clone_20ep.npz \
    /workspace/runs/dagger_clone_vs_clone_20ep.npz \
    /workspace/runs/dagger_iter2_hider_clone_vs_expert_40ep.npz \
    /workspace/runs/dagger_iter2_expert_vs_seeker_clone_40ep.npz \
    /workspace/runs/dagger_iter2_clone_vs_clone_40ep.npz \
    /workspace/runs/dagger_iter2_mixed_20pct_expert_40ep.npz
```

Validated merge evidence:

```text
episodes_collected: 320
samples: 307200
inputs:
- original expert dataset: 100 episodes, 96000 samples
- DAgger iteration 1: 60 episodes, 57600 samples
- DAgger iteration 2: 160 episodes, 153600 samples
```

Two model-capacity checks were run on the same merged dataset.

Large model:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_structured_behavior_clone.py \
    --dataset /workspace/runs/behavioral_dagger_iter2_extensive_merged.npz \
    --out-dir /workspace/runs/bc_structured_dagger_iter2_extensive \
    --epochs 60 \
    --batch-size 2048 \
    --hidden-sizes 512,512,256 \
    --entity-hidden 128 \
    --learning-rate 0.0007 \
    --seed 123
```

Moderate model:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_structured_behavior_clone.py \
    --dataset /workspace/runs/behavioral_dagger_iter2_extensive_merged.npz \
    --out-dir /workspace/runs/bc_structured_dagger_iter2_extensive_moderate \
    --epochs 45 \
    --batch-size 2048 \
    --hidden-sizes 256,256 \
    --entity-hidden 64 \
    --learning-rate 0.001 \
    --seed 123
```

Closed-loop evaluation command:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_clone_vs_pretrained.py \
    --clone-dir /workspace/runs/bc_structured_dagger_iter2_extensive_moderate \
    --episodes 20 \
    --steps 260 \
    --seed 2001 \
    --out /workspace/runs/clone_vs_pretrained_dagger_iter2_extensive_moderate_20/summary.json \
    --report-out /workspace/runs/clone_vs_pretrained_dagger_iter2_extensive_moderate_20/report.txt
```

Best iteration-2 result was the moderate model:

```text
offline validation:
action_exact: 0.105794
movement_exact: 0.132666

closed-loop, 20 episodes per case:
case                            S    H    T    S_win    H_win   T_rate  visible   S_return   H_return
pretrained_vs_pretrained        0   20    0    0.000    1.000    0.000    0.295   -143.900    143.900
clone_hider_vs_pretrained      19    1    0    0.950    0.050    0.000    0.736    113.300   -113.300
pretrained_vs_clone_seeker      1   19    0    0.050    0.950    0.000    0.351   -133.600    133.600
clone_vs_clone                 19    1    0    0.950    0.050    0.000    0.652     66.400    -66.400
```

Conclusion: the extensive DAgger pipeline works, but more feed-forward supervised data is not enough. The hider clone remains the primary failure point and the current single-step model is likely missing temporal state. The next implementation should be a recurrent, role-specific clone trained on sequence chunks and evaluated with hidden state carried across rollout steps.

## Run MAE With Separate Hider And Seeker Models

The evaluator now supports independent role-specific clone checkpoints:

```text
--hider-clone-dir
--seeker-clone-dir
```

This is the current MAE separated-model experiment. It trains one structured clone only on hider actions and one structured clone only on seeker actions from the extensive DAgger dataset.

Train the hider clone:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_structured_behavior_clone.py \
    --dataset /workspace/runs/behavioral_dagger_iter2_extensive_merged.npz \
    --out-dir /workspace/runs/mae_separated_hider_structured_iter2 \
    --role hider \
    --epochs 45 \
    --batch-size 2048 \
    --hidden-sizes 256,256 \
    --entity-hidden 64 \
    --learning-rate 0.001 \
    --seed 4100
```

Train the seeker clone:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/train_structured_behavior_clone.py \
    --dataset /workspace/runs/behavioral_dagger_iter2_extensive_merged.npz \
    --out-dir /workspace/runs/mae_separated_seeker_structured_iter2 \
    --role seeker \
    --epochs 45 \
    --batch-size 2048 \
    --hidden-sizes 256,256 \
    --entity-hidden 64 \
    --learning-rate 0.001 \
    --seed 4200
```

Evaluate both separated models inside MAE:

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/evaluate_clone_vs_pretrained.py \
    --env /workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet \
    --hider-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz \
    --seeker-policy /workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz \
    --hider-clone-dir /workspace/runs/mae_separated_hider_structured_iter2 \
    --seeker-clone-dir /workspace/runs/mae_separated_seeker_structured_iter2 \
    --episodes 20 \
    --steps 260 \
    --seed 2001 \
    --out /workspace/runs/mae_separated_clone_vs_pretrained_20/summary.json \
    --report-out /workspace/runs/mae_separated_clone_vs_pretrained_20/report.txt
```

Validated closed-loop evidence:

```text
case                            S    H    T    S_win    H_win   T_rate  visible   S_return   H_return
pretrained_vs_pretrained        1   19    0    0.050    0.950    0.000    0.317   -132.200    132.200
clone_hider_vs_pretrained      18    2    0    0.900    0.100    0.000    0.747    115.200   -115.200
pretrained_vs_clone_seeker      0   20    0    0.000    1.000    0.000    0.350   -164.300    138.200
clone_vs_clone                 16    4    0    0.800    0.200    0.000    0.598     51.700    -51.700
```

The implementation works, but these role-specific clones are not yet expert-equivalent. The cloned hider remains weak against the pretrained seeker, and the cloned seeker fails against the pretrained hider. Full details are in `docs/mae_separated_models.md`.

## Inspect The Saved Dataset

```bash
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python -c "import numpy as np; data=np.load('/workspace/runs/random_rollouts_hide_seek_quadrant.npz', allow_pickle=True); print('npz_keys', sorted(data.files)); print('episodes', len(data['episodes'])); print('summaries', len(data['summaries'])); print('first_summary', data['summaries'][0])"
```

Validated output from the current workspace:

```text
npz_keys ['episodes', 'manifest', 'summaries']
episodes 10
summaries 10
first_summary {'episode': 0, 'steps': 80, 'total_reward': [-36.0, -36.0, 36.0, 36.0], 'done': True, 'discard_episode': False, ...}
```

## Generate Rollout Videos

The live MuJoCo viewer is useful for inspection, but it depends on OpenGL forwarding. For reproducible visualization, use the saved rollout dataset and generate a top-down animation.

The current collectors store MuJoCo wall geometry as `wall_geoms` in each episode. The renderer uses those wall boxes to draw the actual physical walls. If the render output says `walls_drawn: False`, regenerate the rollout with the current `collect_rollouts.py` or `collect_policy_rollouts.py` script before using the video for collision/debugging analysis.

Generate a GIF for episode 0:

```bash
mkdir -p videos
docker run --rm -v "$PWD:/workspace" mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/render_rollout_video.py \
    --input /workspace/runs/random_rollouts_hide_seek_quadrant.npz \
    --episode 0 \
    --out /workspace/videos/random_rollout_ep0.gif \
    --fps 12
```

Expected output:

```text
input: /workspace/runs/random_rollouts_hide_seek_quadrant.npz
episode: 0
frames: 80
fps: 12
output: /workspace/videos/random_rollout_ep0.gif
walls_drawn: True
```

The generated file on the Linux host is:

```text
./videos/random_rollout_ep0.gif
```

The renderer uses recorded observations and recorded wall geometry only. It does not require MuJoCo rendering, OpenGL, X11, or GPU access.

## Run The Original Viewer On Linux

The original interactive viewer is `multi-agent-emergence-environments/bin/examine.py`.

First, verify that the container sees a valid OpenGL context:

```bash
xhost +local:docker
docker run --rm \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD:/workspace" \
  mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/check_gl.py
```

The viewer requires OpenGL `1.5` or higher.

Then run:

```bash
docker run --rm \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD:/workspace" \
  mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/multi-agent-emergence-environments/bin/examine.py \
    /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet
```

After testing, revoke the broad local X11 permission:

```bash
xhost -local:docker
```

If the viewer opens but appears static, check the current action overlay. The default action is neutral:

```text
action_movement = [5, 5, 5]
```

For `MultiDiscrete([11, 11, 11])`, value `5` is the no-op center value. Use the viewer controls to change actions:

- `Y / U`: select agent
- `G / B`: select action type
- `J / K`: select action dimension
- `A / Z`: decrease/increase selected action
- `N`: reset to next seed
- `Space`: stop/start
- `Right arrow`: step once

## Common Issues

### Docker Permission Denied

If Docker reports permission denied:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Or use `sudo docker ...`.

### Missing Upstream Repositories

If the Docker build fails because files under `mujoco-worldgen` or `multi-agent-emergence-environments` are missing, clone the upstream repos:

```bash
git clone https://github.com/openai/mujoco-worldgen.git
git clone https://github.com/openai/multi-agent-emergence-environments.git
```

### OpenGL Version Too Low

Run:

```bash
docker run --rm \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD:/workspace" \
  mae-legacy:dev \
  conda run --no-capture-output -n mae-legacy \
  python /workspace/scripts/check_gl.py
```

If `GL_VERSION` is lower than `1.5`, use headless rollout collection instead of the live viewer, or run on a Linux machine with working OpenGL/X11 forwarding.

### Slow Viewer

The viewer can be slow if the container uses CPU software rendering. Headless rollout collection is the validated path for reliable data generation.

## Notes On Dependency Choices

The upstream repositories are old. The Docker image intentionally uses a legacy stack rather than modernizing the code:

- Python 3.6 is required for compatibility.
- Native Python 3.12 is not suitable.
- `jsonnet==0.11.2` failed in this setup, so `jsonnet==0.17.0` is used.
- `mujoco-py` requires the legacy MuJoCo 1.50 binaries and key file.

The practical strategy is:

1. Use this Docker image to reproduce and study the original environment.
2. Collect rollouts and establish baseline behavior.
3. Build future training and cybersecurity transfer work in a separate modern stack.
