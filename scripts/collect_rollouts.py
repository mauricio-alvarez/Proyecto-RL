import argparse
import json
import os
import time

import numpy as np

from mujoco_worldgen.util.envs import load_env


def sample_action(env):
    action = env.action_space.sample()
    return {key: np.array(value) for key, value in action.items()}


def serializable_info(info):
    result = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def summarize_obs(obs):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(obs.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="/workspace/runs/random_rollouts_hide_seek_quadrant.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = load_env(
        args.env,
        core_dir="/workspace/multi-agent-emergence-environments",
        envs_dir="mae_envs/envs",
        xmls_dir="xmls",
    )

    first_obs = env.reset()
    manifest = {
        "env": args.env,
        "episodes_requested": args.episodes,
        "max_steps_per_episode": args.steps,
        "seed": args.seed,
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "observation_summary": summarize_obs(first_obs),
        "created_unix": time.time(),
    }

    episodes = []
    summaries = []
    discard_count = 0

    for episode_idx in range(args.episodes):
        obs = env.reset()
        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
        }
        total_reward = None
        done = False
        info = {}

        for step_idx in range(args.steps):
            action = sample_action(env)
            next_obs, reward, done, info = env.step(action)

            episode["observations"].append(obs)
            episode["actions"].append(action)
            episode["rewards"].append(np.array(reward))
            episode["dones"].append(bool(done))
            episode["infos"].append(serializable_info(info))

            total_reward = reward if total_reward is None else total_reward + reward
            obs = next_obs

            if done or info.get("discard_episode"):
                break

        discarded = bool(info.get("discard_episode"))
        discard_count += int(discarded)
        steps = step_idx + 1
        summary = {
            "episode": episode_idx,
            "steps": steps,
            "total_reward": np.asarray(total_reward).round(4).tolist(),
            "done": bool(done),
            "discard_episode": discarded,
            "last_info_keys": sorted(info.keys()),
        }
        summaries.append(summary)
        episodes.append(episode)

        print(
            "episode={episode} steps={steps} total_reward={total_reward} "
            "done={done} discard={discard_episode}".format(**summary)
        )

    np.savez_compressed(
        args.out,
        manifest=np.array(manifest, dtype=object),
        summaries=np.array(summaries, dtype=object),
        episodes=np.array(episodes, dtype=object),
    )

    print("saved:", args.out)
    print("episodes_collected:", len(episodes))
    print("discard_count:", discard_count)
    print("first_observation_keys:", sorted(first_obs.keys()))
    print("manifest_json:", json.dumps(manifest, sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
