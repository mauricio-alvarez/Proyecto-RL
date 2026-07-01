import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

from mujoco_worldgen.util.envs import load_env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_mae_ppo import (  # noqa: E402
    RunningMeanStd,
    build_model,
    evaluate_policy,
    make_serializable,
)


def load_json(path):
    with open(path, "r") as input_file:
        return json.load(input_file)


def write_report(path, summary):
    lines = [
        "MAE PPO Checkpoint Evaluation",
        "env: {}".format(summary["env"]),
        "checkpoint: {}".format(summary["checkpoint"]),
        "episodes: {}".format(summary["episodes"]),
        "seed_start: {}".format(summary["seed"]),
        "",
        "wins:",
        "hiders: {}".format(summary["hider_wins"]),
        "seekers: {}".format(summary["seeker_wins"]),
        "ties: {}".format(summary["ties"]),
        "",
        "rates:",
        "hider_win_rate: {:.4f}".format(summary["hider_win_rate"]),
        "seeker_win_rate: {:.4f}".format(summary["seeker_win_rate"]),
        "tie_rate: {:.4f}".format(summary["tie_rate"]),
        "mean_visible_fraction: {:.4f}".format(summary["mean_visible_fraction"]),
        "",
        "returns:",
        "mean_hider_return: {:.4f}".format(summary["mean_hider_return"]),
        "mean_seeker_return: {:.4f}".format(summary["mean_seeker_return"]),
        "mean_steps: {:.4f}".format(summary["mean_steps"]),
    ]
    with open(path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--train-summary", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--out", default="/workspace/runs/mae_ppo_quadrant_v1/eval_100_summary.json")
    parser.add_argument("--report-out", default="/workspace/runs/mae_ppo_quadrant_v1/eval_100_report.txt")
    args = parser.parse_args()

    train_summary = load_json(args.train_summary)
    normalization = np.load(args.normalization, allow_pickle=True)
    obs_keys = [str(value) for value in normalization["obs_keys"].tolist()]
    roles = normalization["roles"].astype(np.int32)
    n_hiders = int(normalization["n_hiders"])
    n_agents = int(normalization["n_agents"])

    actor_rms = RunningMeanStd(shape=normalization["actor_mean"].shape)
    actor_rms.mean = normalization["actor_mean"].astype(np.float64)
    actor_rms.var = normalization["actor_var"].astype(np.float64)
    actor_rms.count = float(normalization["actor_count"])

    critic_rms = RunningMeanStd(shape=normalization["critic_mean"].shape)
    critic_rms.mean = normalization["critic_mean"].astype(np.float64)
    critic_rms.var = normalization["critic_var"].astype(np.float64)
    critic_rms.count = float(normalization["critic_count"])

    env = load_env(
        args.env,
        core_dir="/workspace/multi-agent-emergence-environments",
        envs_dir="mae_envs/envs",
        xmls_dir="xmls",
    )
    env.seed(args.seed)
    env.reset()

    model = build_model(
        actor_dim=int(train_summary["actor_dim"]),
        critic_dim=int(train_summary["critic_dim"]),
        hidden_sizes=[int(value) for value in train_summary["hidden_sizes"]],
        learning_rate=float(train_summary["learning_rate"]),
        clip_range=float(train_summary["clip_range"]),
        value_coef=float(train_summary["value_coef"]),
        entropy_coef=float(train_summary["entropy_coef"]),
        max_grad_norm=0.5,
    )
    saver = tf.train.Saver()
    max_steps = args.steps or int(getattr(env.unwrapped, "horizon", train_summary["rollout_steps"]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint)
        episodes = evaluate_policy(
            sess=sess,
            env=env,
            model=model,
            actor_rms=actor_rms,
            critic_rms=critic_rms,
            obs_keys=obs_keys,
            roles=roles,
            n_hiders=n_hiders,
            episodes=args.episodes,
            max_steps=max_steps,
            seed=args.seed,
        )

    winners = [episode["winner"] for episode in episodes]
    hider_returns = [episode["hider_mean_return"] for episode in episodes]
    seeker_returns = [episode["seeker_mean_return"] for episode in episodes]
    visible = [episode["visible_fraction"] for episode in episodes]
    steps = [episode["steps"] for episode in episodes]
    summary = {
        "created_unix": time.time(),
        "env": args.env,
        "checkpoint": args.checkpoint,
        "normalization": args.normalization,
        "train_summary": args.train_summary,
        "episodes": len(episodes),
        "seed": args.seed,
        "n_agents": n_agents,
        "n_hiders": n_hiders,
        "n_seekers": n_agents - n_hiders,
        "hider_wins": winners.count("hiders"),
        "seeker_wins": winners.count("seekers"),
        "ties": winners.count("tie"),
        "hider_win_rate": round(float(winners.count("hiders")) / max(len(winners), 1), 4),
        "seeker_win_rate": round(float(winners.count("seekers")) / max(len(winners), 1), 4),
        "tie_rate": round(float(winners.count("tie")) / max(len(winners), 1), 4),
        "mean_hider_return": round(float(np.mean(hider_returns)), 4),
        "mean_seeker_return": round(float(np.mean(seeker_returns)), 4),
        "mean_visible_fraction": round(float(np.mean(visible)), 4),
        "mean_steps": round(float(np.mean(steps)), 4),
        "episodes_detail": episodes,
    }

    for path in [args.out, args.report_out]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as output_file:
        json.dump(make_serializable(summary), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    write_report(args.report_out, summary)

    print("saved_summary:", args.out, flush=True)
    print("saved_report:", args.report_out, flush=True)
    print(
        "result hider_wins={} seeker_wins={} ties={} hider_win_rate={:.4f} seeker_win_rate={:.4f} mean_visible={:.4f}".format(
            summary["hider_wins"],
            summary["seeker_wins"],
            summary["ties"],
            summary["hider_win_rate"],
            summary["seeker_win_rate"],
            summary["mean_visible_fraction"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
