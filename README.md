# Proyecto RL: Multi-Agent Hide-and-Seek Baseline

This workspace sets up the legacy OpenAI multi-agent emergence environments as a reproducible baseline for a seeker/hider reinforcement learning project.

The immediate goal is to reproduce the original hide-and-seek environment, collect rollout data, and use that as the base for later training and transfer experiments, including cybersecurity-inspired seeker/hider environments.

## Current Status

Implemented and validated:

- Cloned `openai/multi-agent-emergence-environments`.
- Cloned required dependency `openai/mujoco-worldgen`.
- Created a legacy Docker image named `mae-legacy:dev`.
- Installed Python 3.6, MuJoCo 1.50, `mujoco-py`, Gym 0.10.8, SciPy 1.3.1, and related dependencies inside Docker.
- Fixed the original `jsonnet==0.11.2` pin by using `jsonnet==0.17.0`, because 0.11.2 failed inside the container.
- Added a headless smoke test.
- Added a rollout collector.
- Collected 10 random-policy episodes from `hide_and_seek_quadrant.jsonnet`.

Not implemented yet:

- RL training loop.
- Saved-policy playback dependencies from `requirements_ma_policy.txt`.
- Verified GUI visualization from Docker on Windows.
- Cybersecurity transfer environment.

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
|   |-- collect_rollouts.py
|   `-- smoke_mae.py
|-- environment.yml
`-- README.md
```

## Prerequisites

Required:

- Windows with Docker Desktop running.
- Docker Desktop configured for Linux containers.
- This project located at:

```text
C:\Users\asus\Documents\Proyecto RL
```

The native Windows Python environment is not used for the working environment. The upstream project is too old for Python 3.12, so the runnable path is Docker.

## Build The Docker Image

From PowerShell, inside the project directory:

```powershell
cd "C:\Users\asus\Documents\Proyecto RL"
docker build -f docker/Dockerfile.mae-legacy -t mae-legacy:dev .
```

Expected result:

```text
naming to docker.io/library/mae-legacy:dev
```

The Dockerfile downloads:

- MuJoCo 1.50 into `/root/.mujoco/mjpro150`
- The public legacy MuJoCo key into `/root/.mujoco/mjkey.txt`
- Python dependencies into Conda environment `mae-legacy`

## Run The Smoke Test

This verifies that MuJoCo, JSONNet, and the hide-and-seek environment work.

```powershell
docker run --rm -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python /workspace/scripts/smoke_mae.py --episodes 3 --steps 20
```

Expected evidence:

```text
action_space: Dict(...)
observation_keys: [...]
episode=0 steps=20 total_reward=[...] done=False discard=False
episode=1 steps=20 total_reward=[...] done=False discard=False
episode=2 steps=20 total_reward=[...] done=False discard=False
```

The `discard=False` value is important. It means the episode did not hit a MuJoCo simulation failure.

## Collect Random Rollouts

This collects observations, actions, rewards, done flags, and info dictionaries.

```powershell
docker run --rm -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python /workspace/scripts/collect_rollouts.py --episodes 10 --steps 120 --seed 7 --out /workspace/runs/random_rollouts_hide_seek_quadrant.npz
```

Validated result:

```text
episodes_collected: 10
discard_count: 0
saved: /workspace/runs/random_rollouts_hide_seek_quadrant.npz
```

The saved file appears on Windows at:

```text
C:\Users\asus\Documents\Proyecto RL\runs\random_rollouts_hide_seek_quadrant.npz
```

## Inspect The Saved Dataset

Use Docker to inspect the `.npz` file:

```powershell
docker run --rm -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python -c "import numpy as np; data=np.load('/workspace/runs/random_rollouts_hide_seek_quadrant.npz', allow_pickle=True); print('npz_keys', sorted(data.files)); print('episodes', len(data['episodes'])); print('summaries', len(data['summaries'])); print('first_summary', data['summaries'][0])"
```

Validated output:

```text
npz_keys ['episodes', 'manifest', 'summaries']
episodes 10
summaries 10
first_summary {'episode': 0, 'steps': 80, 'total_reward': [-36.0, -36.0, 36.0, 36.0], 'done': True, 'discard_episode': False, ...}
```

## Run The Original Examine Script

The original repo provides `bin/examine.py`.

Headless environment loading is already validated through the smoke and rollout scripts. The direct examine command is:

```powershell
docker run --rm -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python /workspace/multi-agent-emergence-environments/bin/examine.py /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet
```

This command may attempt to open an interactive viewer. On Windows Docker Desktop, GUI output from Linux containers requires extra display setup.

## Visualization Options

### Option A: Windows X Server

Install and start an X server such as VcXsrv or X410 on Windows.

For VcXsrv, typical settings:

- Multiple windows
- Display number: `0`
- Start no client
- Disable access control for local development

Then run:

```powershell
docker run --rm -e DISPLAY=host.docker.internal:0.0 -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python /workspace/multi-agent-emergence-environments/bin/examine.py /workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet
```

This path is not yet verified in this workspace. It is the standard direction for GUI forwarding from a Linux Docker container to Windows.

Diagnostic command:

```powershell
docker run --rm -e DISPLAY=host.docker.internal:0.0 -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python /workspace/scripts/check_gl.py
```

The viewer needs an OpenGL context whose version is at least `1.5`.

Previously observed failing VcXsrv result:

```text
DISPLAY: host.docker.internal:0.0
window_created: True
GL_VENDOR: NVIDIA Corporation
GL_RENDERER: NVIDIA GeForce RTX 3060 Ti/PCIe/SSE2
GL_VERSION: 1.4 (4.6.0 NVIDIA 591.86)
```

That fails because MuJoCo reads the context as OpenGL `1.4`.

If this happens, restart VcXsrv and toggle the **Native opengl** option:

- If it was enabled, disable it and rerun `scripts/check_gl.py`.
- If it was disabled, enable it and rerun `scripts/check_gl.py`.

If `GL_VERSION` still starts with `1.4`, VcXsrv is not a viable live-viewer path for this container.

Observed working VcXsrv result:

```text
DISPLAY: host.docker.internal:0.0
window_created: True
GL_VENDOR: Mesa/X.org
GL_RENDERER: llvmpipe (LLVM 11.0.1, 256 bits)
GL_VERSION: 3.1 Mesa 20.3.5
```

This is sufficient for the MuJoCo viewer because the reported OpenGL version is higher than `1.5`.

### Option B: WSLg

Install an Ubuntu WSL distribution and use WSLg for Linux GUI support. This may be cleaner than Windows X-server forwarding, but WSL currently has no distribution installed on this machine.

### Option C: Headless Data First

The currently verified path is headless:

1. Run random rollouts.
2. Save `.npz` trajectory data.
3. Build a separate visualization script that renders trajectories into plots or videos.

This is the most reliable next step if GUI forwarding is unstable.

## Known Technical Notes

- The original project targets Python 3.6.
- Native Windows Python 3.12 cannot install the original dependency stack.
- `jsonnet==0.11.2` failed in Docker with a parser error; `jsonnet==0.17.0` works.
- Random action samples must be converted from tuples to NumPy arrays before stepping the environment. The scripts already handle this.
- Rewards are per-agent arrays. In the quadrant setup with 2 hiders and 2 seekers, rewards look like:

```text
[hider_0, hider_1, seeker_0, seeker_1]
```

## Next Engineering Steps

Recommended order:

1. Add a trajectory-to-video or trajectory-to-plot script.
2. Install and validate saved-policy playback dependencies.
3. Define a minimal training loop for 1 seeker vs 1 hider.
4. Scale to 2 seekers vs 2 hiders.
5. Start a separate modern Gymnasium/PyTorch cybersecurity transfer environment.
