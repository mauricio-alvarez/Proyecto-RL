# Principle-First Cyber Hide-And-Seek

This phase stops trying to clone the MAE neural policies. Instead, it transfers the useful principles:

- seeker/hider asymmetry
- partial discovery
- adversarial search
- visibility and detection
- decoys, hardening, and monitoring
- robust evaluation across randomized environments

The cyber environment is intentionally abstract. It does not execute real exploits, scan real networks, or encode operational intrusion steps. It is a safe simulator for research on adversarial search and defense.

## Environment

Implementation:

- `cyber_rl/env.py`

The environment is a graph. Nodes represent abstract hosts or services. Edges represent reachability. Each episode has:

- entry node
- target node
- hidden vulnerabilities
- node values
- defender patch budget
- defender decoy budget
- detection meter
- episode time limit

The seeker can:

- wait
- scan the current node
- exploit discovered neighboring nodes
- move to compromised neighboring nodes
- exfiltrate at the compromised target

The hider can:

- do nothing
- patch a node
- deploy a decoy
- monitor a node

Terminal outcomes:

- `attacker_success`: seeker compromises and exfiltrates from the target.
- `caught`: detection crosses the threshold or the seeker hits a decoy.
- `timeout`: seeker fails to finish before the time limit.

## Scenario Families

The benchmark uses five graph families:

- `chain`: narrow lateral movement.
- `branching`: tree-like enterprise topology.
- `dense`: many alternate paths.
- `decoy_heavy`: misleading high-vulnerability dead ends.
- `random`: connected random graph.

This is important: a policy is not benchmarked on one hand-picked map.

## Agents

Implemented seekers:

- `random`: random valid actions.
- `greedy`: opportunistic exploitation of high known vulnerabilities.
- `targeted`: shortest-progress seeker toward the target.
- `stealth`: avoids detection and accepts timeout risk.
- `q_table`: tabular Q-learning seeker.
- `linear_q`: linear function-approximation Q-learning seeker.

Implemented hiders:

- `noop`: no defense.
- `random`: random valid defense.
- `patch_high_value`: hardens target and target-neighborhood nodes.
- `decoy_frontier`: places decoys on discovered frontier nodes.
- `adaptive`: monitors when detection rises, otherwise patches and decoys.

## Benchmark

Implementation:

- `scripts/smoke_cyber_rl.py`
- `scripts/run_cyber_benchmarks.py`

Validated benchmark command:

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

Outputs:

- `summary.json`
- `episodes.csv`
- `pair_summaries.csv`
- `report.md`
- `attacker_success.svg`
- `q_training_summary.json`
- `q_table.json`
- `linear_q_training_summary.json`
- `linear_q_weights.json`

## Validated Results

Benchmark v4 ran 7,500 evaluation episodes:

- 5 scenario families
- 50 episodes per family
- 6 seeker policies
- 5 hider policies

Aggregate seeker results:

| seeker | success | caught | timeout | mean return |
|---|---:|---:|---:|---:|
| targeted | 0.399 | 0.601 | 0.000 | -12.47 |
| linear_q | 0.190 | 0.014 | 0.796 | -62.23 |
| stealth | 0.075 | 0.497 | 0.428 | -77.48 |
| greedy | 0.054 | 0.946 | 0.000 | -81.95 |
| random | 0.035 | 0.799 | 0.166 | -89.33 |
| q_table | 0.017 | 0.222 | 0.762 | -92.23 |

Aggregate hider results:

| hider | attacker success | caught | timeout | mean attacker return |
|---|---:|---:|---:|---:|
| decoy_frontier | 0.081 | 0.689 | 0.229 | -81.31 |
| adaptive | 0.085 | 0.639 | 0.276 | -80.00 |
| patch_high_value | 0.105 | 0.492 | 0.403 | -74.76 |
| random | 0.146 | 0.493 | 0.361 | -65.37 |
| noop | 0.224 | 0.253 | 0.523 | -44.96 |

The strongest seeker is `targeted`, a principle-based planner. The strongest hider is `decoy_frontier`, which most reliably suppresses attacker success.

The learned `linear_q` seeker is meaningful but conservative: it has higher success than random, greedy, stealth, and tabular Q in aggregate except for targeted, and it is rarely caught. However, it times out often. This is useful evidence: simple function approximation improves over tabular Q, but the next serious learned seeker should use richer graph-aware function approximation.

## Current Conclusion

This phase delivers a working cyber RL benchmark, not just a concept.

What is solved:

- safe abstract cyber hide-and-seek environment
- seeker and hider action spaces
- randomized scenario families
- hand-coded principle baselines
- tabular and linear Q-learning seekers
- robust benchmark matrix
- machine-readable outputs and plots

What is not solved yet:

- high-performing learned graph policy
- neural multi-agent training
- transfer into a real cyber range

The correct next research step is a graph neural or message-passing policy trained in this simulator, not a return to MAE policy cloning.
