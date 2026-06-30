import argparse
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cyber_rl.benchmark import (  # noqa: E402
    default_attackers,
    default_defenders,
    run_episode,
    summarize_results,
    write_outputs,
)
from cyber_rl.kev import kev_families, load_kev_catalog, make_kev_scenario, normalize_catalog  # noqa: E402
from cyber_rl.q_learning import LinearQAttacker, QTableAttacker  # noqa: E402


def curriculum_defender(episode, episodes, seed):
    defenders = default_defenders()
    phase_size = max(1, int(episodes / 4))
    if episode < phase_size:
        pool = [defenders[0]]
    elif episode < phase_size * 2:
        pool = [defenders[0], defenders[2]]
    elif episode < phase_size * 3:
        pool = [defenders[2], defenders[1]]
    else:
        pool = [defenders[2], defenders[3], defenders[4]]
    return pool[(episode + seed) % len(pool)]


def train_kev_q_table(catalog, episodes, seed, families, n_nodes, max_steps):
    probe = make_kev_scenario(catalog, families[0], seed, n_nodes=n_nodes, max_steps=max_steps)
    agent = QTableAttacker(action_count=3 + 2 * probe.n_nodes, seed=seed + 1)
    agent.name = "q_table_kev"
    history = []

    alpha = 0.15
    gamma = 0.97
    epsilon_start = 0.35
    epsilon_end = 0.05
    for episode in range(episodes):
        family = families[episode % len(families)]
        scenario_seed = seed * 100000 + episode
        scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
        defender = curriculum_defender(episode, episodes, seed)
        env_seed = scenario_seed + 17
        from cyber_rl.env import CyberHideSeekEnv

        env = CyberHideSeekEnv(scenario, seed=env_seed)
        obs = env.reset(seed=env_seed)
        done = False
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - float(episode) / float(max(episodes - 1, 1)))
        while not done:
            defender_action = defender.act(obs)
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action, defender_action)
            agent.update(obs, action, reward, next_obs, done, alpha=alpha, gamma=gamma)
            obs = next_obs
        if (episode + 1) % max(1, episodes // 20) == 0:
            history.append({
                "episode": episode + 1,
                "epsilon": round(epsilon, 4),
                "q_states": len(agent.q),
                "last_outcome": info["outcome"],
                "last_return": info["attacker_return"],
            })

    return agent, {
        "kind": "q_table_kev",
        "episodes": episodes,
        "seed": seed,
        "families": families,
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "q_states": len(agent.q),
        "history": history,
    }


def train_kev_linear_q(catalog, episodes, seed, families, n_nodes, max_steps):
    probe = make_kev_scenario(catalog, families[0], seed, n_nodes=n_nodes, max_steps=max_steps)
    agent = LinearQAttacker(action_count=3 + 2 * probe.n_nodes, n_nodes=n_nodes, max_steps=max_steps, seed=seed + 1)
    agent.name = "linear_q_kev"
    history = []

    alpha = 0.02
    gamma = 0.97
    epsilon_start = 0.45
    epsilon_end = 0.05
    for episode in range(episodes):
        family = families[episode % len(families)]
        scenario_seed = seed * 100000 + episode
        scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
        defender = curriculum_defender(episode, episodes, seed)
        env_seed = scenario_seed + 29
        from cyber_rl.env import CyberHideSeekEnv

        env = CyberHideSeekEnv(scenario, seed=env_seed)
        obs = env.reset(seed=env_seed)
        done = False
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - float(episode) / float(max(episodes - 1, 1)))
        while not done:
            defender_action = defender.act(obs)
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action, defender_action)
            agent.update(obs, action, reward, next_obs, done, alpha=alpha, gamma=gamma)
            obs = next_obs
        if (episode + 1) % max(1, episodes // 20) == 0:
            history.append({
                "episode": episode + 1,
                "epsilon": round(epsilon, 4),
                "last_outcome": info["outcome"],
                "last_return": info["attacker_return"],
                "weight_norm": round(float(np.linalg.norm(agent.weights)), 4),
            })

    return agent, {
        "kind": "linear_q_kev",
        "episodes": episodes,
        "seed": seed,
        "families": families,
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "weight_norm": round(float(np.linalg.norm(agent.weights)), 4),
        "history": history,
    }


def run_kev_suite(catalog, families, episodes_per_family, seed, n_nodes, max_steps, extra_attackers):
    attackers = default_attackers(extra_attackers=extra_attackers)
    defenders = default_defenders()
    rows = []
    pair_summaries = []
    scenario_samples = []

    for attacker in attackers:
        for defender in defenders:
            pair_records = []
            for family_idx, family in enumerate(families):
                for episode_idx in range(episodes_per_family):
                    scenario_seed = seed + family_idx * 10000 + episode_idx
                    eval_seed = seed * 1000000 + family_idx * 10000 + episode_idx
                    scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
                    if len(scenario_samples) < 12:
                        scenario_samples.append(scenario.to_dict())
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
                        "target_cve": scenario.node_metadata[-1].get("cveID"),
                        "target_vendor": scenario.node_metadata[-1].get("vendorProject"),
                        "target_product": scenario.node_metadata[-1].get("product"),
                        "target_ransomware_use": scenario.node_metadata[-1].get("knownRansomwareCampaignUse"),
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
        "dataset": {
            "title": normalized["title"],
            "catalogVersion": normalized["catalogVersion"],
            "dateReleased": normalized["dateReleased"],
            "count": normalized["count"],
            "source_url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
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
        "scenario_samples": scenario_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kev-json", default="/workspace/data/raw/known_exploited_vulnerabilities.json")
    parser.add_argument("--out-dir", default="/workspace/runs/kev_realworld_benchmark_v1")
    parser.add_argument("--episodes-per-family", type=int, default=50)
    parser.add_argument("--q-train-episodes", type=int, default=6000)
    parser.add_argument("--linear-q-train-episodes", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2200)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--families", default=",".join(kev_families()))
    args = parser.parse_args()

    catalog = load_kev_catalog(args.kev_json)
    families = [part.strip() for part in args.families.split(",") if part.strip()]
    random.seed(args.seed)

    q_agent = None
    q_summary = None
    if args.q_train_episodes > 0:
        q_agent, q_summary = train_kev_q_table(
            catalog=catalog,
            episodes=args.q_train_episodes,
            seed=args.seed + 10,
            families=families,
            n_nodes=args.n_nodes,
            max_steps=args.max_steps,
        )

    linear_agent = None
    linear_summary = None
    if args.linear_q_train_episodes > 0:
        linear_agent, linear_summary = train_kev_linear_q(
            catalog=catalog,
            episodes=args.linear_q_train_episodes,
            seed=args.seed + 20,
            families=families,
            n_nodes=args.n_nodes,
            max_steps=args.max_steps,
        )

    extra_attackers = []
    if q_agent is not None:
        extra_attackers.append(q_agent)
    if linear_agent is not None:
        extra_attackers.append(linear_agent)

    suite = run_kev_suite(
        catalog=catalog,
        families=families,
        episodes_per_family=args.episodes_per_family,
        seed=args.seed,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
        extra_attackers=extra_attackers,
    )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    write_outputs(args.out_dir, suite, training_summary=q_summary)
    with open(os.path.join(args.out_dir, "kev_dataset_summary.json"), "w") as handle:
        json.dump(suite["dataset"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(args.out_dir, "kev_scenario_samples.json"), "w") as handle:
        json.dump(suite["scenario_samples"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    if q_summary is not None:
        with open(os.path.join(args.out_dir, "q_table_kev_training_summary.json"), "w") as handle:
            json.dump(q_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(args.out_dir, "q_table_kev.json"), "w") as handle:
            json.dump(q_agent.to_jsonable(), handle)
            handle.write("\n")
    if linear_summary is not None:
        with open(os.path.join(args.out_dir, "linear_q_kev_training_summary.json"), "w") as handle:
            json.dump(linear_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(args.out_dir, "linear_q_kev_weights.json"), "w") as handle:
            json.dump(linear_agent.to_jsonable(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    print("dataset_title:", suite["dataset"]["title"])
    print("catalogVersion:", suite["dataset"]["catalogVersion"])
    print("dateReleased:", suite["dataset"]["dateReleased"])
    print("dataset_count:", suite["dataset"]["count"])
    print("saved_dir:", args.out_dir)
    print("episodes:", len(suite["rows"]))
    print("pair_summaries:", len(suite["pair_summaries"]))
    for row in sorted(suite["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        print(
            "attacker={attacker} success={attacker_success_rate:.3f} "
            "caught={caught_rate:.3f} timeout={timeout_rate:.3f} "
            "return={mean_attacker_return:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
