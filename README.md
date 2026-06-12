# Proyecto RL: Linux Reproduction Guide

This repository contains a reproducible Docker setup for the legacy OpenAI multi-agent emergence environments. It is intended as the baseline for a seeker/hider reinforcement learning project.

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
- OpenGL diagnostic script in `scripts/check_gl.py`.
- Methodology and validation notes in `docs/`.
- A sample random rollout artifact in `runs/random_rollouts_hide_seek_quadrant.npz`.

Not implemented:

- RL training loop.
- Saved-policy playback dependencies from `requirements_ma_policy.txt`.
- Cybersecurity transfer environment.
- Production-quality trajectory visualization.

## Repository Layout

```text
.
|-- docker/
|   `-- Dockerfile.mae-legacy
|-- docs/
|   |-- environment_validation.md
|   |-- methodology.md
|   `-- rollout_collection.md
|-- mujoco-worldgen/
|-- multi-agent-emergence-environments/
|-- runs/
|   `-- random_rollouts_hide_seek_quadrant.npz
|-- scripts/
|   |-- check_gl.py
|   |-- collect_rollouts.py
|   `-- smoke_mae.py
|-- ASIS.md
|-- environment.yml
`-- README.md
```

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
