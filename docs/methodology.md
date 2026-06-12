# Multi-Agent Hide-and-Seek RL Methodology

## Objective

Train two classes of agents, seekers and hiders, in simulated multi-agent environments inspired by `openai/multi-agent-emergence-environments`, then transfer the learned behaviors to security-oriented environments where seekers discover weaknesses and hiders model concealment, deception, or defensive hardening.

The training base should follow the upstream design: start with `Base`, add environment modules such as agents, walls, boxes, ramps, food, or construction objects, and implement rewards/observations/game mechanics through wrappers. This keeps the project extensible when moving from physical hide-and-seek to abstract cybersecurity tasks.

## Phase 1: Reproduce The Base Environment

Goal: make the original hide-and-seek environment run deterministically.

Tasks:
- Clone and install `mujoco-worldgen` and `multi-agent-emergence-environments`.
- Launch `bin/examine.py examples/hide_and_seek_quadrant.jsonnet`.
- Launch a saved policy with `examples/hide_and_seek_quadrant.npz`.
- Record environment reset, observation keys, action spaces, reward ranges, and episode stats.

Acceptance criteria:
- The environment resets without MuJoCo errors.
- Random policy rollouts complete at least 10 episodes.
- Saved policy playback works.
- We can export rollout traces: observations, actions, rewards, done flags, and info metrics.

## Phase 2: Baseline Multi-Agent Training

Goal: produce reproducible seeker/hider policies before introducing transfer.

Recommended baseline:
- Multi-agent PPO with centralized training and decentralized execution.
- Parameter sharing within each team: one hider policy shared by all hiders, one seeker policy shared by all seekers.
- Team reward variants from the upstream environment: `selfish`, `joint_mean`, and `joint_zero_sum`.
- Curriculum over environment complexity.

Curriculum:
1. Empty arena: seekers learn pursuit, hiders learn evasion.
2. Quadrant rooms: introduce walls, doors, and line-of-sight.
3. Object interaction: boxes, ramps, grabbing, locking.
4. Random rooms: generalize across layouts.
5. Transfer tasks: box locking, shelter construction, blueprint construction.

Core metrics:
- Seeker success rate: percentage of time any seeker sees a hider.
- Hider survival rate: percentage of episode hidden from all seekers.
- Time to first detection.
- Object use: box/ramp movement distance, lock frequency, prep-phase changes.
- Generalization: zero-shot performance on unseen seeds, object counts, room counts, and maps.
- Emergence indicators: repeated tool-use motifs, division of labor, blocking, fort construction, pursuit patterns.

## Phase 3: Representation Extraction

Goal: extract reusable functions from trained agents rather than only exporting raw policies.

Artifacts to extract:
- Perception encoders: line-of-sight, local occupancy, object affordances, teammate/opponent state.
- Option-like skills: pursue, evade, block, lock, explore, inspect, hide, coordinate.
- Belief/state summaries: "where can the opponent see me?", "what object changes visibility?", "which region is unexplored?"
- Team communication signals, if communication is added later.

Experiments:
- Freeze perception encoder, retrain policy head on new layouts.
- Freeze low-level movement, retrain high-level option selector.
- Distill seeker and hider policies into interpretable behavior classifiers.
- Train probes to predict hidden variables: opponent visibility, blocked paths, exploitable openings.

## Phase 4: Cybersecurity Transfer Environment

Goal: map hide-and-seek dynamics onto controlled security simulations.

Cybersecurity mapping:
- Seeker: vulnerability finder, scanner, red-team explorer, penetration-testing agent.
- Hider: defender, patching/hardening agent, deceptive service, attacker evasion model, or vulnerable component that tries to avoid detection depending on scenario.
- Walls/rooms: network segments, hosts, services, permissions, firewall rules.
- Boxes/ramps/tools: credentials, exploits, scanners, logs, patches, honeypots, privilege-escalation paths.
- Visibility: observability through scans, logs, exposed ports, dependency graphs, SBOMs, runtime telemetry.
- Locking/blocking: patching, access-control changes, service isolation, credential rotation.

Initial security environments:
1. Toy graph network: nodes are hosts/services, edges are connectivity, vulnerabilities are hidden node attributes.
2. Capture-the-flag lab: seeker must find vulnerable path; hider changes configuration within constraints.
3. Web app dependency graph: seeker searches vulnerable packages/configs; hider patches or hides signals.
4. Blue-team/red-team co-training: seeker finds exploit chains while hider minimizes successful compromise and operational cost.

Transfer experiments:
- Zero-shot: evaluate physical-environment seeker representation on graph exploration after a new output head.
- Few-shot: fine-tune with limited security episodes.
- Ablation: train security agents from scratch and compare sample efficiency.
- Curriculum transfer: gradually replace geometric observations with graph/security observations.
- Robustness: evaluate against unseen vulnerability distributions and deceptive signals.

## Phase 5: Evaluation And Safety

Use only sandboxed, synthetic, or explicitly authorized targets.

Security metrics:
- Vulnerability discovery rate.
- False-positive and false-negative rates.
- Exploit-chain depth found.
- Mean time to detection.
- Defensive cost: number of patches, blocked services, broken dependencies.
- Generalization to unseen topologies and vulnerability classes.

Safety constraints:
- No real external scanning.
- No exploit execution outside local labs.
- All generated payloads stay inside intentionally vulnerable test systems.
- Log all actions and keep deterministic seeds for auditability.

## Suggested First Milestone

Milestone 0 is environment validation:
- Make the original MuJoCo hide-and-seek example launch.
- Run 10 random-policy episodes.
- Run one saved policy playback.
- Save a short rollout dataset.

Milestone 1 is a minimal training loop:
- Train 1 hider vs 1 seeker in a small quadrant environment.
- Scale to 2 hiders vs 2 seekers.
- Compare `selfish`, `joint_mean`, and `joint_zero_sum` rewards.

Milestone 2 is transfer:
- Build a small graph-security Gym environment.
- Reuse the seeker/hider interface and reward framing.
- Test representation transfer against scratch training.

