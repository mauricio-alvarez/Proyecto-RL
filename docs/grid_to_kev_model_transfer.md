# Grid Hide-And-Seek To KEV Cyber Transfer

This experiment implements the project path where models are trained in a game environment first, then translated into the cybersecurity environment.

The implementation does not reuse MAE checkpoint weights. It trains new local models on two simple hide-and-seek worlds, then transfers their learned concept encoder weights into KEV cyber attacker/defender models.

## Why This Counts As Model Transfer

The grid game and the cyber benchmark share a concept vector of size `32`.

Grid concepts include:

- relative seeker/hider position
- distance to opponent
- line-of-sight
- obstacle/cover features
- time pressure
- role identity

Cyber concepts map the same slots to analogous structure:

- attacker distance to target
- discovered and compromised fractions
- detection pressure
- frontier size
- patch/decoy budget
- target value
- role identity

The trained grid seeker and hider encoders initialize the cyber attacker and defender encoders:

```text
grid seeker encoder -> cyber attacker encoder
grid hider encoder  -> cyber defender encoder
```

The cyber output heads are new because cyber actions are not grid movement actions.

## Implementation

Main script:

- `scripts/train_grid_to_kev_transfer.py`

Trained artifacts:

- `runs/grid_to_kev_transfer_v1/grid_seeker.pt`
- `runs/grid_to_kev_transfer_v1/grid_hider.pt`
- `runs/grid_to_kev_transfer_v1/transfer_attacker.pt`
- `runs/grid_to_kev_transfer_v1/transfer_defender.pt`
- `runs/grid_to_kev_transfer_v1/scratch_attacker.pt`
- `runs/grid_to_kev_transfer_v1/scratch_defender.pt`

Reports:

- `runs/grid_to_kev_transfer_v1/transfer_report.md`
- `runs/grid_to_kev_transfer_v1/transfer_summary.json`
- `runs/grid_to_kev_transfer_v1/cyber_transfer_pair_summaries.csv`
- `runs/grid_to_kev_transfer_v1/cyber_transfer_episodes.csv`

## GPU Runtime

The legacy MAE Docker image is CPU-only for TensorFlow and has no PyTorch installed. GPU training uses:

```text
pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

Validated GPU:

```text
NVIDIA GeForce RTX 3060 Ti
PyTorch 2.4.1+cu124
CUDA available: True
```

## Reproduce

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

## Results

Grid training:

| model | validation accuracy |
|---|---:|
| seeker | 0.998 |
| hider | 0.989 |

Grid concept probes:

| role | environment | opponent | result |
|---|---|---|---|
| seeker | empty | random hider | 97.3% catch |
| seeker | wall | random hider | 99.3% catch |
| seeker | empty | expert hider | 0% catch, mean final distance 2.41 |
| seeker | wall | expert hider | 0% catch, mean final distance 1.00 |
| hider | empty | random seeker | 100% survival |
| hider | wall | random seeker | 100% survival |
| hider | empty | expert seeker | 100% survival, mean final distance 2.42 |
| hider | wall | expert seeker | 100% survival, mean final distance 1.00 |

Cyber imitation validation:

| cyber role | transfer init | scratch |
|---|---:|---:|
| attacker | 0.7354 | 0.7409 |
| defender | 0.7569 | 0.7613 |

Closed-loop KEV benchmark:

| attacker | success | caught | timeout | mean return |
|---|---:|---:|---:|---:|
| targeted | 0.161 | 0.839 | 0.000 | -63.40 |
| transfer_attacker | 0.103 | 0.870 | 0.027 | -75.51 |
| scratch_attacker | 0.101 | 0.867 | 0.031 | -75.73 |

Defender comparison:

| defender | attacker success | caught | timeout | attacker return |
|---|---:|---:|---:|---:|
| adaptive | 0.095 | 0.840 | 0.065 | -75.21 |
| decoy_frontier | 0.097 | 0.901 | 0.002 | -78.05 |
| transfer_defender | 0.137 | 0.855 | 0.008 | -68.89 |
| scratch_defender | 0.158 | 0.838 | 0.004 | -64.04 |

## Interpretation

The trained game models learned the basic concepts:

- seeker: pursue and reduce distance to target/hider
- hider: evade, maintain distance, and use cover/obstacle structure

The transferred cyber attacker is slightly better than the scratch neural attacker in closed-loop KEV evaluation:

```text
transfer_attacker success: 0.1029
scratch_attacker success:  0.1014
```

The transferred cyber defender is more clearly useful than the scratch defender:

```text
attacker success vs transfer_defender: 0.1371
attacker success vs scratch_defender:  0.1581
```

This means the transferred hider/defender representation did carry useful defensive structure into the cyber domain.

The result is not yet a state-of-the-art cyber RL model. The strongest hand-coded baseline remains `targeted` for attackers and `decoy_frontier`/`adaptive` for defenders. But the requested pipeline is now implemented:

1. Train hider/seeker models in simple game environments.
2. Transfer their encoders to cyber attacker/defender models.
3. Fine-tune on KEV-derived cyber behavior.
4. Benchmark transfer models against scratch models and principle baselines.
5. Save trained artifacts and reproducible reports.

## Next Upgrade

The next improvement should use a graph neural policy over KEV scenario topology. The current transfer works through fixed concept vectors; graph neural transfer would preserve more topology information and likely improve cyber generalization.
