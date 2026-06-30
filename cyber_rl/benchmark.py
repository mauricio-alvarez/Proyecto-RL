import csv
import json
import math
import os
import time

from cyber_rl.env import CyberHideSeekEnv, make_scenario
from cyber_rl.policies import (
    AdaptiveDefender,
    DecoyFrontierDefender,
    GreedyAttacker,
    NoopDefender,
    PatchHighValueDefender,
    RandomAttacker,
    RandomDefender,
    StealthAttacker,
    TargetedAttacker,
)


DEFAULT_FAMILIES = ["chain", "branching", "dense", "decoy_heavy", "random"]


def mean(values):
    return sum(values) / float(max(len(values), 1))


def stderr_binary(rate, n):
    if n <= 1:
        return 0.0
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / float(n))


def run_episode(scenario, attacker, defender, seed):
    env = CyberHideSeekEnv(scenario, seed=seed)
    obs = env.reset(seed=seed)
    attacker.reset()
    defender.reset()
    done = False
    while not done:
        defender_action = defender.act(obs)
        attacker_action = attacker.act(obs)
        obs, reward, done, info = env.step(attacker_action, defender_action)
    return info


def summarize_results(records):
    n = len(records)
    success_rate = mean([1.0 if row["attacker_success"] else 0.0 for row in records])
    caught_rate = mean([1.0 if row["caught"] else 0.0 for row in records])
    timeout_rate = mean([1.0 if row["timeout"] else 0.0 for row in records])
    return {
        "episodes": n,
        "attacker_success_rate": round(success_rate, 4),
        "attacker_success_stderr": round(stderr_binary(success_rate, n), 4),
        "caught_rate": round(caught_rate, 4),
        "timeout_rate": round(timeout_rate, 4),
        "mean_steps": round(mean([row["steps"] for row in records]), 4),
        "mean_attacker_return": round(mean([row["attacker_return"] for row in records]), 4),
        "mean_detection": round(mean([row["detection"] for row in records]), 4),
        "mean_compromised_count": round(mean([row["compromised_count"] for row in records]), 4),
    }


def default_attackers(extra_attackers=None):
    attackers = [
        RandomAttacker(seed=101),
        GreedyAttacker(),
        TargetedAttacker(),
        StealthAttacker(),
    ]
    if extra_attackers:
        attackers.extend(extra_attackers)
    return attackers


def default_defenders():
    return [
        NoopDefender(),
        RandomDefender(seed=202),
        PatchHighValueDefender(),
        DecoyFrontierDefender(),
        AdaptiveDefender(),
    ]


def run_benchmark_suite(
    episodes_per_family=30,
    seed=900,
    families=None,
    n_nodes=8,
    max_steps=24,
    q_attacker=None,
    extra_attackers=None,
):
    if families is None:
        families = DEFAULT_FAMILIES
    extras = []
    if q_attacker is not None:
        extras.append(q_attacker)
    if extra_attackers:
        extras.extend(extra_attackers)
    attackers = default_attackers(extra_attackers=extras)
    defenders = default_defenders()
    rows = []
    summaries = []

    for attacker in attackers:
        for defender in defenders:
            pair_records = []
            for family_idx, family in enumerate(families):
                for episode_idx in range(int(episodes_per_family)):
                    scenario_seed = seed + family_idx * 10000 + episode_idx
                    eval_seed = seed * 1000000 + family_idx * 10000 + episode_idx
                    scenario = make_scenario(family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
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
                        "scans": info["event_counts"]["scans"],
                        "exploit_successes": info["event_counts"]["exploit_successes"],
                        "exploit_failures": info["event_counts"]["exploit_failures"],
                    }
                    rows.append(row)
                    pair_records.append(row)
            summary = summarize_results(pair_records)
            summary.update({
                "attacker": attacker.name,
                "defender": defender.name,
                "families": list(families),
            })
            summaries.append(summary)

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

    return {
        "created_unix": time.time(),
        "episodes_per_family": int(episodes_per_family),
        "families": list(families),
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "seed": seed,
        "rows": rows,
        "pair_summaries": summaries,
        "aggregate_by_attacker": aggregate_by_attacker,
        "aggregate_by_defender": aggregate_by_defender,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_report(path, suite, training_summary=None):
    lines = []
    lines.append("# Cyber RL Benchmark Report")
    lines.append("")
    lines.append("- episodes_per_family: {}".format(suite["episodes_per_family"]))
    lines.append("- families: {}".format(", ".join(suite["families"])))
    lines.append("- n_nodes: {}".format(suite["n_nodes"]))
    lines.append("- max_steps: {}".format(suite["max_steps"]))
    lines.append("")
    if training_summary is not None:
        lines.append("## Q-Learning Training")
        lines.append("")
        lines.append("- episodes: {}".format(training_summary["episodes"]))
        lines.append("- q_states: {}".format(training_summary["q_states"]))
        lines.append("- alpha: {}".format(training_summary["alpha"]))
        lines.append("- gamma: {}".format(training_summary["gamma"]))
        lines.append("")
    lines.append("## Aggregate By Attacker")
    lines.append("")
    lines.append("| attacker | success | caught | timeout | return | steps |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(suite["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        lines.append("| {attacker} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} | {mean_steps:.2f} |".format(**row))
    lines.append("")
    lines.append("## Aggregate By Defender")
    lines.append("")
    lines.append("| defender | attacker success | caught | timeout | return | steps |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(suite["aggregate_by_defender"], key=lambda item: item["attacker_success_rate"]):
        lines.append("| {defender} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} | {mean_steps:.2f} |".format(**row))
    lines.append("")
    lines.append("## Pair Summary")
    lines.append("")
    lines.append("| attacker | defender | success | stderr | caught | timeout | return |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(suite["pair_summaries"], key=lambda item: (item["attacker"], item["defender"])):
        lines.append("| {attacker} | {defender} | {attacker_success_rate:.3f} | {attacker_success_stderr:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} |".format(**row))
    lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def write_success_svg(path, suite):
    rows = sorted(suite["aggregate_by_attacker"], key=lambda item: item["attacker"])
    width = 760
    height = 260
    margin_left = 140
    bar_height = 28
    gap = 18
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="30" font-family="Arial" font-size="18" font-weight="700">Attacker success rate by policy</text>',
    ]
    max_bar = width - margin_left - 80
    y = 58
    for row in rows:
        rate = row["attacker_success_rate"]
        bar_width = int(max_bar * rate)
        lines.append('<text x="20" y="{}" font-family="Arial" font-size="13">{}</text>'.format(y + 19, row["attacker"]))
        lines.append('<rect x="{}" y="{}" width="{}" height="{}" fill="#2f6f8f"/>'.format(margin_left, y, bar_width, bar_height))
        lines.append('<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#cccccc"/>'.format(margin_left, y, max_bar, bar_height))
        lines.append('<text x="{}" y="{}" font-family="Arial" font-size="13">{:.1f}%</text>'.format(margin_left + max_bar + 12, y + 19, rate * 100.0))
        y += bar_height + gap
    lines.append("</svg>")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def write_outputs(output_dir, suite, training_summary=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_json(os.path.join(output_dir, "summary.json"), suite)
    write_csv(os.path.join(output_dir, "episodes.csv"), suite["rows"])
    write_csv(os.path.join(output_dir, "pair_summaries.csv"), suite["pair_summaries"])
    write_markdown_report(os.path.join(output_dir, "report.md"), suite, training_summary=training_summary)
    write_success_svg(os.path.join(output_dir, "attacker_success.svg"), suite)
