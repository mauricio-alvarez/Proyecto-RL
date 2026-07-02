import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cyber_rl.benchmark import default_defenders, run_episode, summarize_results  # noqa: E402
from cyber_rl.env import (  # noqa: E402
    ATTACK_EXFILTRATE,
    ATTACK_EXPLOIT_OFFSET,
    ATTACK_SCAN,
    ATTACK_WAIT,
    DEFEND_NOOP,
)
from cyber_rl.kev import kev_families, load_kev_catalog, make_kev_scenario, normalize_catalog  # noqa: E402
from cyber_rl.policies import (  # noqa: E402
    AdaptiveDefender,
    DecoyFrontierDefender,
    GreedyAttacker,
    NoopDefender,
    PatchHighValueDefender,
    RandomAttacker,
    RandomDefender,
    StealthAttacker,
    TargetedAttacker,
    valid_indices,
)
from cyber_rl.q_learning import LinearQAttacker, QTableAttacker  # noqa: E402
from train_mae_ppo import RunningMeanStd, build_model, make_serializable  # noqa: E402


ROLE_HIDER = 0
ROLE_SEEKER = 1


def load_json(path):
    with open(path, "r") as input_file:
        return json.load(input_file)


def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.maximum(np.sum(exp), 1e-8)


def load_linear_q(path, seed=0):
    payload = load_json(path)
    agent = LinearQAttacker(
        action_count=payload["action_count"],
        n_nodes=payload["n_nodes"],
        max_steps=payload.get("max_steps", 24),
        seed=seed,
    )
    agent.weights = np.asarray(payload["weights"], dtype=np.float32)
    agent.name = "linear_q_kev"
    return agent


def load_q_table(path, seed=0):
    agent = QTableAttacker.from_jsonable(load_json(path), seed=seed)
    agent.name = "q_table_kev"
    return agent


def cyber_concept_vector(obs, role_id, actor_dim):
    n_nodes = len(obs["adjacency"])
    n = float(max(n_nodes, 1))
    current = obs["current_node"]
    target = obs["target_node"]
    known_current = obs["known_vuln"][current]
    if known_current < 0:
        known_current = 0
    degrees = [len(neighbors) / n for neighbors in obs["adjacency"]]

    features = [
        float(role_id),
        float(current) / n,
        float(target) / n,
        min(float(obs["dist_to_target"][current]) / n, 1.5),
        float(obs["steps_remaining"]) / 24.0,
        min(float(obs["detection"]) / 6.0, 1.5),
        float(sum(obs["discovered"])) / n,
        float(sum(obs["compromised"])) / n,
        float(sum(obs["scanned"])) / n,
        float(sum(obs["patched"])) / n,
        float(sum(obs["decoys"])) / n,
        float(obs["patch_budget_left"]) / 3.0,
        float(obs["decoy_budget_left"]) / 2.0,
        min(float(obs["values"][target]) / 10.0, 1.5),
        min(float(obs["values"][current]) / 10.0, 1.5),
        float(known_current) / 5.0,
    ]

    for node in range(n_nodes):
        known = obs["known_vuln"][node]
        if known < 0:
            known = 0
        features.extend([
            1.0 if obs["discovered"][node] else 0.0,
            1.0 if obs["compromised"][node] else 0.0,
            1.0 if obs["scanned"][node] else 0.0,
            float(known) / 5.0,
            1.0 if obs["patched"][node] else 0.0,
            1.0 if obs["decoys"][node] else 0.0,
            min(float(obs["values"][node]) / 10.0, 1.5),
            min(float(obs["dist_to_target"][node]) / n, 1.5),
            1.0 if node in obs["adjacency"][current] else 0.0,
            1.0 if node == target else 0.0,
        ])

    features.extend(degrees)
    features.extend([
        min(float(max(obs["values"]) if obs["values"] else 0.0) / 10.0, 1.5),
        min(float(min(obs["values"]) if obs["values"] else 0.0) / 10.0, 1.5),
        float(len(valid_indices(obs["valid_attacker_actions"]))) / float(max(len(obs["valid_attacker_actions"]), 1)),
        float(len(valid_indices(obs["valid_defender_actions"]))) / float(max(len(obs["valid_defender_actions"]), 1)),
    ])

    vector = np.zeros((actor_dim,), dtype=np.float32)
    usable = min(actor_dim, len(features))
    vector[:usable] = np.asarray(features[:usable], dtype=np.float32)
    return vector.reshape(1, -1)


class MAEPPOProbe(object):
    def __init__(self, checkpoint, normalization_path, summary_path):
        self.checkpoint = checkpoint
        self.normalization_path = normalization_path
        self.summary = load_json(summary_path)
        normalization = np.load(normalization_path, allow_pickle=True)

        self.actor_rms = RunningMeanStd(shape=normalization["actor_mean"].shape)
        self.actor_rms.mean = normalization["actor_mean"].astype(np.float64)
        self.actor_rms.var = normalization["actor_var"].astype(np.float64)
        self.actor_rms.count = float(normalization["actor_count"])

        self.actor_dim = int(self.summary["actor_dim"])
        self.critic_dim = int(self.summary["critic_dim"])
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model = build_model(
                actor_dim=self.actor_dim,
                critic_dim=self.critic_dim,
                hidden_sizes=[int(value) for value in self.summary["hidden_sizes"]],
                learning_rate=float(self.summary["learning_rate"]),
                clip_range=float(self.summary["clip_range"]),
                value_coef=float(self.summary["value_coef"]),
                entropy_coef=float(self.summary["entropy_coef"]),
                max_grad_norm=0.5,
                reset_graph=False,
            )
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())
            self.saver.restore(self.session, checkpoint)

    def policy_stats(self, obs, role_id):
        vector = cyber_concept_vector(obs, role_id, self.actor_dim)
        actor_norm = self.actor_rms.normalize(vector)
        logits = self.session.run(
            self.model["logits"],
            feed_dict={
                self.model["actor_x"]: actor_norm,
                self.model["role"]: np.asarray([role_id], dtype=np.int32),
            },
        )
        probs = {key: softmax(value[0]) for key, value in logits.items()}
        movement_expectation = []
        movement_up = []
        movement_down = []
        for key in ["movement_0", "movement_1", "movement_2"]:
            classes = np.arange(len(probs[key]), dtype=np.float32)
            movement_expectation.append(float(np.sum(classes * probs[key])))
            movement_up.append(float(np.sum(probs[key][6:])))
            movement_down.append(float(np.sum(probs[key][:5])))
        return {
            "movement_expectation": movement_expectation,
            "movement_up": movement_up,
            "movement_down": movement_down,
            "pull": float(probs["pull"][1]),
            "glue": float(probs["glueall"][1]),
        }


class MAETransferAttacker(object):
    name = "mae_ppo_transfer_attacker"

    def __init__(self, probe):
        self.probe = probe

    def reset(self):
        return None

    def act(self, obs):
        valid = obs["valid_attacker_actions"]
        valid_actions = valid_indices(valid)
        if not valid_actions:
            return ATTACK_WAIT
        if valid[ATTACK_EXFILTRATE]:
            return ATTACK_EXFILTRATE

        stats = self.probe.policy_stats(obs, ROLE_SEEKER)
        n_nodes = len(obs["adjacency"])
        current = obs["current_node"]
        target = obs["target_node"]
        detection_pressure = min(float(obs["detection"]) / 6.0, 1.5)
        pursuit = np.mean(stats["movement_up"]) - np.mean(stats["movement_down"])
        exploit_drive = stats["pull"]
        caution = stats["glue"]

        if valid[ATTACK_WAIT] and detection_pressure > 0.85 and caution > exploit_drive:
            return ATTACK_WAIT
        if valid[ATTACK_SCAN] and not obs["scanned"][current]:
            return ATTACK_SCAN

        scored = []
        for action in valid_actions:
            score = -1000.0
            if action == ATTACK_WAIT:
                score = -2.0 + 4.0 * caution * detection_pressure
            elif action == ATTACK_SCAN:
                unknown_neighbors = sum(1 for node in obs["adjacency"][current] if not obs["discovered"][node])
                score = 1.0 + unknown_neighbors - 1.5 * detection_pressure
            elif ATTACK_EXPLOIT_OFFSET <= action < ATTACK_EXPLOIT_OFFSET + n_nodes:
                node = action - ATTACK_EXPLOIT_OFFSET
                known = obs["known_vuln"][node]
                if known < 0:
                    known = 1
                progress = max(0, obs["dist_to_target"][current] - obs["dist_to_target"][node])
                score = (
                    2.0
                    + 4.0 * exploit_drive
                    + 1.5 * pursuit
                    + 2.0 * progress
                    + 0.8 * float(known)
                    + 0.6 * float(obs["values"][node])
                    - 3.0 * detection_pressure
                )
            elif ATTACK_EXPLOIT_OFFSET + n_nodes <= action < ATTACK_EXPLOIT_OFFSET + 2 * n_nodes:
                node = action - ATTACK_EXPLOIT_OFFSET - n_nodes
                progress = obs["dist_to_target"][current] - obs["dist_to_target"][node]
                score = 1.0 + 3.0 * pursuit + 2.5 * progress + 0.5 * obs["values"][node] - detection_pressure
            if action == ATTACK_EXFILTRATE:
                score = 100.0
            scored.append((score, -action, action))
        scored.sort(reverse=True)
        return scored[0][-1]


class MAETransferDefender(object):
    name = "mae_ppo_transfer_defender"

    def __init__(self, probe):
        self.probe = probe

    def reset(self):
        return None

    def act(self, obs):
        valid = obs["valid_defender_actions"]
        valid_actions = valid_indices(valid)
        if not valid_actions:
            return DEFEND_NOOP

        stats = self.probe.policy_stats(obs, ROLE_HIDER)
        n_nodes = len(obs["adjacency"])
        current = obs["current_node"]
        target = obs["target_node"]
        concealment = stats["glue"]
        intervention = stats["pull"]
        evasion = np.mean(stats["movement_down"]) - np.mean(stats["movement_up"])
        detection_pressure = min(float(obs["detection"]) / 6.0, 1.5)

        scored = []
        for action in valid_actions:
            score = 0.0
            if action == DEFEND_NOOP:
                score = -0.5 + 1.0 * evasion - detection_pressure
            elif 1 <= action < 1 + n_nodes:
                node = action - 1
                near_target = 1.0 / (1.0 + obs["dist_to_target"][node])
                score = (
                    1.0
                    + 3.0 * intervention
                    + 2.0 * near_target
                    + 0.8 * obs["values"][node]
                    + (2.0 if node == target else 0.0)
                    - (3.0 if obs["patched"][node] else 0.0)
                )
            elif 1 + n_nodes <= action < 1 + 2 * n_nodes:
                node = action - 1 - n_nodes
                frontier = 1.0 if obs["discovered"][node] and not obs["compromised"][node] else 0.0
                score = (
                    1.0
                    + 4.0 * concealment
                    + 2.0 * frontier
                    + 1.0 / (1.0 + obs["dist_to_target"][node])
                    - (4.0 if obs["decoys"][node] else 0.0)
                    - (3.0 if node in (obs["entry_node"], target) else 0.0)
                )
            elif 1 + 2 * n_nodes <= action < 1 + 3 * n_nodes:
                node = action - 1 - 2 * n_nodes
                score = (
                    1.0
                    + 3.0 * detection_pressure
                    + 2.0 * intervention
                    + (2.0 if node == current else 0.0)
                    + 1.0 / (1.0 + obs["dist_to_target"][node])
                )
            scored.append((score, -action, action))
        scored.sort(reverse=True)
        return scored[0][-1]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    with open(path, "w") as output_file:
        json.dump(make_serializable(payload), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def write_report(path, suite):
    lines = [
        "# MAE PPO To KEV Transfer Evaluation",
        "",
        "- transfer_type: zero-shot concept adapter",
        "- episodes_per_family: {}".format(suite["episodes_per_family"]),
        "- families: {}".format(", ".join(suite["families"])),
        "- n_nodes: {}".format(suite["n_nodes"]),
        "- max_steps: {}".format(suite["max_steps"]),
        "",
        "## Aggregate By Attacker",
        "",
        "| attacker | success | caught | timeout | return | steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(suite["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        lines.append("| {attacker} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} | {mean_steps:.2f} |".format(**row))
    lines.extend([
        "",
        "## Aggregate By Defender",
        "",
        "| defender | attacker success | caught | timeout | return | steps |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(suite["aggregate_by_defender"], key=lambda item: item["attacker_success_rate"]):
        lines.append("| {defender} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} | {mean_steps:.2f} |".format(**row))
    lines.extend([
        "",
        "## Pair Summary",
        "",
        "| attacker | defender | success | caught | timeout | return |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(suite["pair_summaries"], key=lambda item: (item["attacker"], item["defender"])):
        lines.append("| {attacker} | {defender} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} |".format(**row))
    with open(path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def run_suite(catalog, attackers, defenders, families, episodes_per_family, seed, n_nodes, max_steps):
    rows = []
    pair_summaries = []
    for attacker in attackers:
        for defender in defenders:
            pair_records = []
            for family_idx, family in enumerate(families):
                for episode_idx in range(episodes_per_family):
                    scenario_seed = seed + family_idx * 10000 + episode_idx
                    eval_seed = seed * 1000000 + family_idx * 10000 + episode_idx
                    scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
                    info = run_episode(scenario, attacker, defender, eval_seed)
                    row = {
                        "attacker": attacker.name,
                        "defender": defender.name,
                        "family": family,
                        "episode": episode_idx,
                        "seed": scenario_seed,
                        "outcome": info["outcome"],
                        "attacker_success": int(info["attacker_success"]),
                        "caught": int(info["caught"]),
                        "timeout": int(info["timeout"]),
                        "steps": info["steps"],
                        "attacker_return": info["attacker_return"],
                        "detection": info["detection"],
                        "compromised_count": info["compromised_count"],
                    }
                    rows.append(row)
                    pair_records.append(row)
            summary = summarize_results(pair_records)
            summary.update({
                "attacker": attacker.name,
                "defender": defender.name,
                "families": list(families),
            })
            pair_summaries.append(summary)

    aggregate_by_attacker = []
    for attacker_name in sorted(set(row["attacker"] for row in rows)):
        records = [row for row in rows if row["attacker"] == attacker_name]
        summary = summarize_results(records)
        summary["attacker"] = attacker_name
        aggregate_by_attacker.append(summary)

    aggregate_by_defender = []
    for defender_name in sorted(set(row["defender"] for row in rows)):
        records = [row for row in rows if row["defender"] == defender_name]
        summary = summarize_results(records)
        summary["defender"] = defender_name
        aggregate_by_defender.append(summary)

    normalized = normalize_catalog(catalog)
    return {
        "created_unix": time.time(),
        "dataset": {
            "title": normalized["title"],
            "catalogVersion": normalized["catalogVersion"],
            "dateReleased": normalized["dateReleased"],
            "count": normalized["count"],
        },
        "episodes_per_family": episodes_per_family,
        "families": list(families),
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "seed": seed,
        "rows": rows,
        "pair_summaries": pair_summaries,
        "aggregate_by_attacker": aggregate_by_attacker,
        "aggregate_by_defender": aggregate_by_defender,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kev-json", default="/workspace/data/raw/known_exploited_vulnerabilities.json")
    parser.add_argument("--checkpoint", default="/workspace/runs/mae_ppo_quadrant_v1/model.ckpt-300")
    parser.add_argument("--normalization", default="/workspace/runs/mae_ppo_quadrant_v1/normalization.npz")
    parser.add_argument("--train-summary", default="/workspace/runs/mae_ppo_quadrant_v1/summary.json")
    parser.add_argument("--q-table-json", default="/workspace/runs/kev_realworld_benchmark_v1/q_table_kev.json")
    parser.add_argument("--linear-q-json", default="/workspace/runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json")
    parser.add_argument("--out-dir", default="/workspace/runs/mae_ppo_kev_transfer_v1")
    parser.add_argument("--episodes-per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--families", default=",".join(kev_families()))
    parser.add_argument(
        "--exclude-mae-defender",
        action="store_true",
        help="Evaluate the transferred MAE attacker only against default cyber defenders. Useful for single-role checkpoints.",
    )
    args = parser.parse_args()

    catalog = load_kev_catalog(args.kev_json)
    families = [part.strip() for part in args.families.split(",") if part.strip()]
    probe = MAEPPOProbe(args.checkpoint, args.normalization, args.train_summary)
    mae_attacker = MAETransferAttacker(probe)
    mae_defender = MAETransferDefender(probe)

    attackers = [
        RandomAttacker(seed=101),
        GreedyAttacker(),
        TargetedAttacker(),
        StealthAttacker(),
        load_q_table(args.q_table_json, seed=args.seed + 1),
        load_linear_q(args.linear_q_json, seed=args.seed + 2),
        mae_attacker,
    ]
    defenders = default_defenders()
    if not args.exclude_mae_defender:
        defenders = defenders + [mae_defender]

    suite = run_suite(
        catalog=catalog,
        attackers=attackers,
        defenders=defenders,
        families=families,
        episodes_per_family=args.episodes_per_family,
        seed=args.seed,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
    )
    suite["transfer"] = {
        "type": "zero_shot_mae_ppo_concept_adapter",
        "checkpoint": args.checkpoint,
        "normalization": args.normalization,
        "train_summary": args.train_summary,
        "q_table_json": args.q_table_json,
        "linear_q_json": args.linear_q_json,
        "exclude_mae_defender": args.exclude_mae_defender,
    }

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    write_json(os.path.join(args.out_dir, "summary.json"), suite)
    write_csv(os.path.join(args.out_dir, "episodes.csv"), suite["rows"])
    write_csv(os.path.join(args.out_dir, "pair_summaries.csv"), suite["pair_summaries"])
    write_report(os.path.join(args.out_dir, "report.md"), suite)

    print("saved_dir:", args.out_dir, flush=True)
    print("episodes:", len(suite["rows"]), flush=True)
    for row in sorted(suite["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        print(
            "attacker={attacker} success={attacker_success_rate:.3f} caught={caught_rate:.3f} timeout={timeout_rate:.3f} return={mean_attacker_return:.2f}".format(**row),
            flush=True,
        )


if __name__ == "__main__":
    main()
