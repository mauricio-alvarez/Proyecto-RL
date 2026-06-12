# Environment Validation

Validation date: 2026-06-08

## What Was Done

Cloned:
- `openai/multi-agent-emergence-environments`
- `openai/mujoco-worldgen`

Created a local virtual environment:

```powershell
& 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
```

Installed local repos in editable mode:

```powershell
.\.venv\Scripts\python.exe -m pip install setuptools wheel
.\.venv\Scripts\python.exe -m pip install --use-pep517 --no-build-isolation --no-deps --config-settings editable_mode=compat -e .\mujoco-worldgen -e .\multi-agent-emergence-environments
```

Result: editable package installation succeeded.

## Docker Validation

Docker Desktop 4.77.0 with Linux containers is working from this project when Docker commands are run with elevated access.

Built the image:

```powershell
docker build -f docker/Dockerfile.mae-legacy -t mae-legacy:dev .
```

Important setup changes:
- The image installs Python 3.6 through Conda.
- The image downloads MuJoCo 1.50 into `/root/.mujoco/mjpro150`.
- The image downloads the public unlocked legacy MuJoCo key into `/root/.mujoco/mjkey.txt`.
- `jsonnet` was updated from the upstream pin `0.11.2` to `0.17.0` because `0.11.2` failed to parse even a minimal snippet in the container.

Smoke test:

```powershell
docker run --rm -v "C:\Users\asus\Documents\Proyecto RL:/workspace" mae-legacy:dev conda run --no-capture-output -n mae-legacy python /workspace/scripts/smoke_mae.py --episodes 3 --steps 20
```

Result:
- `mujoco_py` imports successfully.
- JSONNet config loading works.
- `hide_and_seek_quadrant.jsonnet` resets successfully.
- Random actions can step the environment when sampled tuple actions are converted to NumPy arrays.
- 3 short random rollouts completed without MuJoCo discard episodes.

Longer rollout collection also works. See `docs/rollout_collection.md`.

## Native Windows Blocker

The available Python runtime is Python 3.12.13. The upstream project states it was tested on Python 3.6, and `mujoco-worldgen/requirements.txt` pins old packages:

```text
scipy==1.3.1
gym==0.10.8
jsonnet==0.11.2
mujoco-py<1.50.2,>=1.50.1
```

Installing the upstream requirements failed under Python 3.12 because `scipy==1.3.1` tries to build `numpy==1.14.5`, which is incompatible with the current interpreter/toolchain.

WSL has no Linux distribution installed.

## Conclusion

The cloned repository structure is usable, and the original environment now runs in Docker. The native Python 3.12 Windows venv remains unsuitable for this upstream stack.

## Recommended Next Setup Path

Preferred:
- Continue using Docker Desktop with Linux containers.
- Use `mae-legacy:dev` for reproduction and rollout collection.
- Keep modern training/transfer code separate from the legacy reproduction container.

Alternative:
- Install an Ubuntu WSL distribution.
- Install Miniconda/Mambaforge inside WSL.
- Create the environment from `environment.yml`.

Smoke test once the legacy runtime is available:

```bash
python multi-agent-emergence-environments/bin/examine.py \
  multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet
```

Saved policy playback:

```bash
python multi-agent-emergence-environments/bin/examine.py \
  multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet \
  multi-agent-emergence-environments/examples/hide_and_seek_quadrant.npz
```
