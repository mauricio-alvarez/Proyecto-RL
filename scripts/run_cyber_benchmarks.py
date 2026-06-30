import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cyber_rl.benchmark import DEFAULT_FAMILIES, run_benchmark_suite, write_outputs
from cyber_rl.q_learning import train_linear_q_attacker, train_q_attacker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/runs/cyber_principle_benchmark")
    parser.add_argument("--episodes-per-family", type=int, default=30)
    parser.add_argument("--q-train-episodes", type=int, default=4000)
    parser.add_argument("--linear-q-train-episodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    args = parser.parse_args()

    families = [part.strip() for part in args.families.split(",") if part.strip()]
    q_agent = None
    training_summary = None
    linear_q_agent = None
    linear_training_summary = None
    if args.q_train_episodes > 0:
        q_agent, training_summary = train_q_attacker(
            episodes=args.q_train_episodes,
            seed=args.seed + 100,
            families=families,
            n_nodes=args.n_nodes,
            max_steps=args.max_steps,
        )
    if args.linear_q_train_episodes > 0:
        linear_q_agent, linear_training_summary = train_linear_q_attacker(
            episodes=args.linear_q_train_episodes,
            seed=args.seed + 200,
            families=families,
            n_nodes=args.n_nodes,
            max_steps=args.max_steps,
        )

    suite = run_benchmark_suite(
        episodes_per_family=args.episodes_per_family,
        seed=args.seed,
        families=families,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
        q_attacker=q_agent,
        extra_attackers=[linear_q_agent] if linear_q_agent is not None else None,
    )
    report_training_summary = training_summary
    if linear_training_summary is not None:
        report_training_summary = dict(training_summary or {})
        report_training_summary["linear_q"] = linear_training_summary
    write_outputs(args.out_dir, suite, training_summary=report_training_summary)

    if training_summary is not None:
        with open(os.path.join(args.out_dir, "q_training_summary.json"), "w") as handle:
            json.dump(training_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(args.out_dir, "q_table.json"), "w") as handle:
            json.dump(q_agent.to_jsonable(), handle)
            handle.write("\n")
    if linear_training_summary is not None:
        with open(os.path.join(args.out_dir, "linear_q_training_summary.json"), "w") as handle:
            json.dump(linear_training_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(args.out_dir, "linear_q_weights.json"), "w") as handle:
            json.dump(linear_q_agent.to_jsonable(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    print("saved_dir:", args.out_dir)
    print("episodes:", len(suite["rows"]))
    print("pair_summaries:", len(suite["pair_summaries"]))
    for row in sorted(suite["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        print(
            "attacker={attacker} success={attacker_success_rate:.3f} "
            "caught={caught_rate:.3f} return={mean_attacker_return:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
