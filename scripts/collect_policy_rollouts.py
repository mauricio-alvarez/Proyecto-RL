import argparse
import json
import os
import time

import numpy as np

from ma_policy.load_policy import load_policy
from mujoco_worldgen.util.envs import load_env


def serializable_info(info):
    result = {}
    for key, value in info.items():
        result[key] = make_serializable(value)
    return result


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


def capture_layout(env):
    base_env = env.unwrapped
    placement_grid = getattr(base_env, "placement_grid", None)
    metadata = getattr(base_env, "metadata", {})
    layout = {
        "current_seed": make_serializable(getattr(base_env, "current_seed", None)),
        "metadata": make_serializable(metadata),
    }
    if placement_grid is not None:
        layout["placement_grid"] = np.asarray(placement_grid, dtype=np.int8)
        layout["grid_size"] = int(placement_grid.shape[0])
    if "floor_size" in metadata:
        layout["floor_size"] = float(metadata["floor_size"])
    elif hasattr(base_env, "floor_size"):
        layout["floor_size"] = float(base_env.floor_size)
    layout["wall_geoms"] = capture_wall_geoms(base_env)
    return layout


def capture_wall_geoms(base_env):
    sim = getattr(base_env, "sim", None)
    if sim is None:
        return []

    wall_geoms = []
    model = sim.model
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id)
        if name is None or not name.startswith("wall"):
            continue
        wall_geoms.append({
            "name": name,
            "pos": sim.data.geom_xpos[geom_id, :3].copy().tolist(),
            "size": model.geom_size[geom_id, :3].copy().tolist(),
            "rgba": model.geom_rgba[geom_id].copy().tolist(),
        })
    return wall_geoms


def summarize_obs(obs):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(obs.items())
    }


def summarize_policy_info(policy_info):
    summary = {}
    for key, value in policy_info.items():
        if isinstance(value, dict):
            summary[key] = {
                sub_key: {
                    "shape": list(sub_value.shape),
                    "dtype": str(sub_value.dtype),
                }
                for sub_key, sub_value in value.items()
                if isinstance(sub_value, np.ndarray)
            }
        elif isinstance(value, np.ndarray):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "mean": float(np.mean(value)),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet",
    )
    parser.add_argument(
        "--policy",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.npz",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="/workspace/runs/policy_rollouts_hide_seek_quadrant.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env = load_env(
        args.env,
        core_dir="/workspace/multi-agent-emergence-environments",
        envs_dir="mae_envs/envs",
        xmls_dir="xmls",
    )
    env.seed(args.seed)

    policy = load_policy(args.policy, env=env, scope="policy_0")
    policy.reset()

    first_obs = env.reset()
    manifest = {
        "env": args.env,
        "policy": args.policy,
        "episodes_requested": args.episodes,
        "max_steps_per_episode": args.steps,
        "seed": args.seed,
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "observation_summary": summarize_obs(first_obs),
        "initial_layout": capture_layout(env),
        "created_unix": time.time(),
        "collector": "collect_policy_rollouts.py",
    }

    episodes = []
    summaries = []
    discard_count = 0

    for episode_idx in range(args.episodes):
        env.seed(args.seed + episode_idx)
        obs = env.reset()
        policy.reset()

        episode = {
            "layout": capture_layout(env),
            "observations": [],
            "actions": [],
            "policy_infos": [],
            "rewards": [],
            "dones": [],
            "infos": [],
        }
        total_reward = None
        done = False
        info = {}
        policy_info = {}

        for step_idx in range(args.steps):
            action, policy_info = policy.act(obs)
            next_obs, reward, done, info = env.step(action)

            episode["observations"].append(obs)
            episode["actions"].append(action)
            episode["policy_infos"].append(summarize_policy_info(policy_info))
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
            "seed": args.seed + episode_idx,
            "steps": steps,
            "total_reward": np.asarray(total_reward).round(4).tolist(),
            "done": bool(done),
            "discard_episode": discarded,
            "last_info_keys": sorted(info.keys()),
        }
        summaries.append(summary)
        episodes.append(episode)

        print(
            "episode={episode} seed={seed} steps={steps} "
            "total_reward={total_reward} done={done} discard={discard_episode}".format(**summary)
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
    print("last_policy_info:", json.dumps(summarize_policy_info(policy_info), sort_keys=True)[:2000])
    print("manifest_json:", json.dumps(make_serializable(manifest), sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
