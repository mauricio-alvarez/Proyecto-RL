import argparse
import json
import os
import time

import numpy as np

from ma_policy.load_policy import load_policy
from mujoco_worldgen.util.envs import load_env


def make_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: make_serializable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(sub_value) for sub_value in value]
    return value


def split_team_obs(obs, n_hiders):
    hider_obs = {key: value[:n_hiders] for key, value in obs.items()}
    seeker_obs = {key: value[n_hiders:] for key, value in obs.items()}
    return hider_obs, seeker_obs


def combine_team_actions(hider_action, seeker_action):
    return {
        key: np.concatenate([hider_action[key], seeker_action[key]], axis=0)
        for key in hider_action
    }


def visible_hider_mask(obs, n_hiders):
    if "mask_aa_obs" not in obs:
        return None
    return np.asarray(obs["mask_aa_obs"])[n_hiders:, :n_hiders].astype(bool)


def evaluate_episode(env, hider_policy, seeker_policy, seed, max_steps, n_hiders):
    env.seed(seed)
    obs = env.reset()
    hider_policy.reset()
    seeker_policy.reset()

    total_reward = None
    visible_steps = 0
    first_visible_step = None
    done = False
    info = {}

    for step_idx in range(max_steps):
        visible_mask = visible_hider_mask(obs, n_hiders)
        if visible_mask is not None and np.any(visible_mask):
            visible_steps += 1
            if first_visible_step is None:
                first_visible_step = step_idx

        hider_obs, seeker_obs = split_team_obs(obs, n_hiders)
        hider_action, _ = hider_policy.act(hider_obs)
        seeker_action, _ = seeker_policy.act(seeker_obs)
        action = combine_team_actions(hider_action, seeker_action)

        obs, reward, done, info = env.step(action)
        total_reward = reward if total_reward is None else total_reward + reward
        if done or info.get("discard_episode"):
            break

    steps = step_idx + 1
    total_reward = np.asarray(total_reward)
    n_agents = total_reward.shape[0]
    n_seekers = n_agents - n_hiders
    hider_return = total_reward[:n_hiders]
    seeker_return = total_reward[n_hiders:]
    hider_mean = float(np.mean(hider_return))
    seeker_mean = float(np.mean(seeker_return))

    if seeker_mean > hider_mean:
        winner = "seekers"
    elif hider_mean > seeker_mean:
        winner = "hiders"
    else:
        winner = "tie"

    return {
        "seed": seed,
        "steps": steps,
        "done": bool(done),
        "discard_episode": bool(info.get("discard_episode")),
        "total_reward": total_reward.round(4).tolist(),
        "hider_mean_return": round(hider_mean, 4),
        "seeker_mean_return": round(seeker_mean, 4),
        "winner": winner,
        "first_visible_step": first_visible_step,
        "visible_fraction": round(float(visible_steps) / float(steps), 4),
        "n_hiders": n_hiders,
        "n_seekers": n_seekers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet",
    )
    parser.add_argument("--hider-policy", required=True)
    parser.add_argument("--seeker-policy", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--n-hiders", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    env = load_env(
        args.env,
        core_dir="/workspace/multi-agent-emergence-environments",
        envs_dir="mae_envs/envs",
        xmls_dir="xmls",
    )
    env.seed(args.seed)
    env.reset()

    n_hiders = args.n_hiders or int(env.metadata.get("n_hiders", 0))
    if n_hiders <= 0:
        raise ValueError("Could not determine n_hiders; pass --n-hiders explicitly")

    hider_policy = load_policy(args.hider_policy, env=env, scope="hider_policy")
    seeker_policy = load_policy(args.seeker_policy, env=env, scope="seeker_policy")

    episodes = []
    for episode_idx in range(args.episodes):
        result = evaluate_episode(
            env=env,
            hider_policy=hider_policy,
            seeker_policy=seeker_policy,
            seed=args.seed + episode_idx,
            max_steps=args.steps,
            n_hiders=n_hiders,
        )
        episodes.append(result)
        print(
            "episode={idx} seed={seed} steps={steps} winner={winner} "
            "hider_mean={hider_mean_return} seeker_mean={seeker_mean_return} "
            "first_visible_step={first_visible_step} visible_fraction={visible_fraction}".format(
                idx=episode_idx,
                **result
            )
        )

    winners = [episode["winner"] for episode in episodes]
    summary = {
        "env": args.env,
        "hider_policy": args.hider_policy,
        "seeker_policy": args.seeker_policy,
        "episodes": len(episodes),
        "seed_start": args.seed,
        "created_unix": time.time(),
        "hider_wins": winners.count("hiders"),
        "seeker_wins": winners.count("seekers"),
        "ties": winners.count("tie"),
        "mean_hider_return": round(float(np.mean([e["hider_mean_return"] for e in episodes])), 4),
        "mean_seeker_return": round(float(np.mean([e["seeker_mean_return"] for e in episodes])), 4),
        "mean_visible_fraction": round(float(np.mean([e["visible_fraction"] for e in episodes])), 4),
        "episodes_detail": episodes,
    }

    print("summary_json:", json.dumps(make_serializable(summary), sort_keys=True))

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as output_file:
            json.dump(make_serializable(summary), output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print("saved:", args.out)


if __name__ == "__main__":
    main()
