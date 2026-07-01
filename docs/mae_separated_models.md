# MAE Separated Hider And Seeker Models

This experiment runs the original MAE hide-and-seek environment with two independent learned models:

- one hider clone checkpoint
- one seeker clone checkpoint

The models are structured behavior clones trained from the MAE DAgger dataset. They are not from-scratch multi-agent RL policies. The purpose is to split the policy interface by role and test whether hider and seeker models can be trained, loaded, and evaluated independently inside MAE.

## Environment

Validated MAE environment:

```text
multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet
```

Expert policies used for labels and closed-loop comparison:

```text
hider expert:  multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz
seeker expert: multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz
```

Training dataset:

```text
runs/behavioral_dagger_iter2_extensive_merged.npz
```

That dataset contains 307,200 samples from the original expert dataset plus DAgger correction rollouts.

## Code Changes

Main evaluator:

```text
scripts/evaluate_clone_vs_pretrained.py
```

The evaluator now supports:

```text
--hider-clone-dir
--seeker-clone-dir
```

When the two directories differ, each clone is loaded into its own TensorFlow graph and session. This avoids tensor-name collisions from importing two independent TensorFlow 1.x checkpoint meta-graphs into the same default graph.

Backward compatibility remains available through:

```text
--clone-dir
```

## Train The Hider Model

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

Artifacts:

```text
runs/mae_separated_hider_structured_iter2/model.ckpt.*
runs/mae_separated_hider_structured_iter2/preprocessing.npz
runs/mae_separated_hider_structured_iter2/metrics.json
runs/mae_separated_hider_structured_iter2/metrics.txt
```

Validated hider offline metrics:

```text
validation action_exact:   0.069466
validation movement_exact: 0.086751
validation movement_0:     0.341764
validation movement_1:     0.418164
validation movement_2:     0.374121
validation pull:           0.765462
validation glueall:        0.757943
```

## Train The Seeker Model

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

Artifacts:

```text
runs/mae_separated_seeker_structured_iter2/model.ckpt.*
runs/mae_separated_seeker_structured_iter2/preprocessing.npz
runs/mae_separated_seeker_structured_iter2/metrics.json
runs/mae_separated_seeker_structured_iter2/metrics.txt
```

Validated seeker offline metrics:

```text
validation action_exact:   0.137500
validation movement_exact: 0.178743
validation movement_0:     0.418652
validation movement_1:     0.536263
validation movement_2:     0.563802
validation pull:           0.786003
validation glueall:        0.753353
```

## Evaluate The Two Separate Models In MAE

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

Outputs:

```text
runs/mae_separated_clone_vs_pretrained_20/summary.json
runs/mae_separated_clone_vs_pretrained_20/report.txt
```

Validated closed-loop result:

```text
case                            S    H    T    S_win    H_win   T_rate  visible   S_return   H_return
----------------------------------------------------------------------------------------------------------------
pretrained_vs_pretrained        1   19    0    0.050    0.950    0.000    0.317   -132.200    132.200
clone_hider_vs_pretrained      18    2    0    0.900    0.100    0.000    0.747    115.200   -115.200
pretrained_vs_clone_seeker      0   20    0    0.000    1.000    0.000    0.350   -164.300    138.200
clone_vs_clone                 16    4    0    0.800    0.200    0.000    0.598     51.700    -51.700
```

`S` is seeker wins, `H` is hider wins, and `T` is ties.

## Interpretation

The separated hider/seeker implementation works technically:

- Two role-specific checkpoints are trained.
- Two independent checkpoints can be loaded in the same evaluation process.
- The MAE environment can run closed-loop with cloned hiders, cloned seekers, or both.

The separated models are not expert-equivalent yet:

- The cloned hider wins only 2/20 episodes against the pretrained seeker.
- The cloned seeker wins 0/20 episodes against the pretrained hider.
- Clone-vs-clone produces seeker wins, but that is not enough evidence because both cloned roles are weaker than their pretrained counterparts.

The main behavioral gap is still closed-loop robustness, not basic loading or offline imitation. A better next model should carry temporal state across rollout steps, for example a role-specific recurrent clone trained on sequence chunks from the DAgger dataset.
