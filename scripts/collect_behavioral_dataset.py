import argparse
import json
import os
import time

import numpy as np

from ma_policy.load_policy import load_policy
from mujoco_worldgen.util.envs import load_env


ROLE_HIDER = 0
ROLE_SEEKER = 1
WINNER_HIDERS = 0
WINNER_SEEKERS = 1
WINNER_TIE = 2


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


def agent_visible_flags(obs, n_hiders, n_agents):
    mask = visible_hider_mask(obs, n_hiders)
    flags = np.zeros(n_agents, dtype=np.int8)
    if mask is None:
        return flags

    for hider_idx in range(n_hiders):
        flags[hider_idx] = int(np.any(mask[:, hider_idx]))
    for seeker_offset in range(n_agents - n_hiders):
        flags[n_hiders + seeker_offset] = int(np.any(mask[seeker_offset, :]))
    return flags


def summarize_obs(obs):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(obs.items())
    }


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


def extend_sample_buffers(
    buffers,
    obs,
    next_obs,
    action,
    reward,
    done,
    visible_flags,
    episode_idx,
    seed,
    step_idx,
    n_hiders,
):
    n_agents = len(reward)
    for agent_idx in range(n_agents):
        role = ROLE_HIDER if agent_idx < n_hiders else ROLE_SEEKER
        policy_id = role
        buffers["episode"].append(episode_idx)
        buffers["seed"].append(seed)
        buffers["step"].append(step_idx)
        buffers["agent_index"].append(agent_idx)
        buffers["role"].append(role)
        buffers["policy_id"].append(policy_id)
        buffers["reward"].append(float(reward[agent_idx]))
        buffers["done"].append(int(done))
        buffers["visible_any"].append(int(visible_flags[agent_idx]))

        for key, value in obs.items():
            buffers["obs"][key].append(np.asarray(value[agent_idx]).copy())
        for key, value in next_obs.items():
            buffers["next_obs"][key].append(np.asarray(value[agent_idx]).copy())
        for key, value in action.items():
            buffers["action"][key].append(np.asarray(value[agent_idx]).copy())


def new_buffers(obs_keys, action_keys):
    return {
        "episode": [],
        "seed": [],
        "step": [],
        "agent_index": [],
        "role": [],
        "policy_id": [],
        "reward": [],
        "done": [],
        "visible_any": [],
        "episode_winner": [],
        "obs": {key: [] for key in obs_keys},
        "next_obs": {key: [] for key in obs_keys},
        "action": {key: [] for key in action_keys},
    }


def append_episode_winner(buffers, episode_sample_count, winner_id):
    buffers["episode_winner"].extend([winner_id] * episode_sample_count)


def stack_buffers(buffers):
    arrays = {
        "episode": np.asarray(buffers["episode"], dtype=np.int32),
        "seed": np.asarray(buffers["seed"], dtype=np.int32),
        "step": np.asarray(buffers["step"], dtype=np.int32),
        "agent_index": np.asarray(buffers["agent_index"], dtype=np.int16),
        "role": np.asarray(buffers["role"], dtype=np.int8),
        "policy_id": np.asarray(buffers["policy_id"], dtype=np.int8),
        "reward": np.asarray(buffers["reward"], dtype=np.float32),
        "done": np.asarray(buffers["done"], dtype=np.int8),
        "visible_any": np.asarray(buffers["visible_any"], dtype=np.int8),
        "episode_winner": np.asarray(buffers["episode_winner"], dtype=np.int8),
    }
    for key, values in buffers["obs"].items():
        arrays["obs_{}".format(key)] = np.asarray(values)
    for key, values in buffers["next_obs"].items():
        arrays["next_obs_{}".format(key)] = np.asarray(values)
    for key, values in buffers["action"].items():
        arrays["action_{}".format(key)] = np.asarray(values)
    return arrays


def winner_from_returns(total_reward, n_hiders):
    hider_mean = float(np.mean(total_reward[:n_hiders]))
    seeker_mean = float(np.mean(total_reward[n_hiders:]))
    if seeker_mean > hider_mean:
        return WINNER_SEEKERS, "seekers", hider_mean, seeker_mean
    if hider_mean > seeker_mean:
        return WINNER_HIDERS, "hiders", hider_mean, seeker_mean
    return WINNER_TIE, "tie", hider_mean, seeker_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet",
    )
    parser.add_argument(
        "--hider-policy",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/d_ramp_defense.npz",
    )
    parser.add_argument(
        "--seeker-policy",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_policy_phases/c_ramps.npz",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n-hiders", type=int, default=0)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--out", default="/workspace/runs/behavioral_hide_seek_full.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env = load_env(
        args.env,
        core_dir="/workspace/multi-agent-emergence-environments",
        envs_dir="mae_envs/envs",
        xmls_dir="xmls",
    )
    env.seed(args.seed)
    first_obs = env.reset()

    metadata = getattr(env, "metadata", getattr(env.unwrapped, "metadata", {}))
    n_hiders = args.n_hiders or int(metadata.get("n_hiders", 0))
    if n_hiders <= 0:
        raise ValueError("Could not determine n_hiders; pass --n-hiders explicitly")

    hider_policy = load_policy(args.hider_policy, env=env, scope="hider_policy", stochastic=args.stochastic_policy)
    seeker_policy = load_policy(args.seeker_policy, env=env, scope="seeker_policy", stochastic=args.stochastic_policy)

    hider_obs, seeker_obs = split_team_obs(first_obs, n_hiders)
    hider_action, _ = hider_policy.act(hider_obs)
    seeker_action, _ = seeker_policy.act(seeker_obs)
    first_action = combine_team_actions(hider_action, seeker_action)
    buffers = new_buffers(sorted(first_obs.keys()), sorted(first_action.keys()))

    manifest = {
        "schema_version": "behavioral_dataset_v1",
        "collector": "collect_behavioral_dataset.py",
        "created_unix": time.time(),
        "env": args.env,
        "hider_policy": args.hider_policy,
        "seeker_policy": args.seeker_policy,
        "stochastic_policy": args.stochastic_policy,
        "episodes_requested": args.episodes,
        "max_steps_per_episode": args.steps,
        "seed": args.seed,
        "n_hiders": n_hiders,
        "role_encoding": {"hider": ROLE_HIDER, "seeker": ROLE_SEEKER},
        "winner_encoding": {"hiders": WINNER_HIDERS, "seekers": WINNER_SEEKERS, "tie": WINNER_TIE},
        "policy_id_encoding": {"hider_policy": ROLE_HIDER, "seeker_policy": ROLE_SEEKER},
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "observation_summary": summarize_obs(first_obs),
        "sample_axis": "one row per episode-step-agent",
        "notes": [
            "obs_* contains the pre-action observation for one agent.",
            "action_* contains the policy action taken from that observation.",
            "next_obs_* contains the post-step observation for the same agent.",
            "visible_any is role-aware: hider visible to any seeker, or seeker sees any hider.",
        ],
    }

    episode_summaries = []
    episode_layouts = []
    discard_count = 0

    for episode_idx in range(args.episodes):
        seed = args.seed + episode_idx
        env.seed(seed)
        obs = env.reset()
        hider_policy.reset()
        seeker_policy.reset()

        episode_layouts.append(capture_layout(env))
        start_count = len(buffers["episode"])
        total_reward = None
        visible_steps = 0
        first_visible_step = None
        done = False
        info = {}

        for step_idx in range(args.steps):
            visible_flags = agent_visible_flags(obs, n_hiders, len(next(iter(obs.values()))))
            if np.any(visible_flags):
                visible_steps += 1
                if first_visible_step is None:
                    first_visible_step = step_idx

            hider_obs, seeker_obs = split_team_obs(obs, n_hiders)
            hider_action, _ = hider_policy.act(hider_obs)
            seeker_action, _ = seeker_policy.act(seeker_obs)
            action = combine_team_actions(hider_action, seeker_action)

            next_obs, reward, done, info = env.step(action)
            reward = np.asarray(reward)
            total_reward = reward if total_reward is None else total_reward + reward

            extend_sample_buffers(
                buffers=buffers,
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                visible_flags=visible_flags,
                episode_idx=episode_idx,
                seed=seed,
                step_idx=step_idx,
                n_hiders=n_hiders,
            )

            obs = next_obs
            if done or info.get("discard_episode"):
                break

        episode_sample_count = len(buffers["episode"]) - start_count
        total_reward = np.asarray(total_reward)
        winner_id, winner_name, hider_mean, seeker_mean = winner_from_returns(total_reward, n_hiders)
        append_episode_winner(buffers, episode_sample_count, winner_id)

        discarded = bool(info.get("discard_episode"))
        discard_count += int(discarded)
        steps = step_idx + 1
        summary = {
            "episode": episode_idx,
            "seed": seed,
            "steps": steps,
            "samples": episode_sample_count,
            "done": bool(done),
            "discard_episode": discarded,
            "winner": winner_name,
            "winner_id": winner_id,
            "total_reward": total_reward.round(4).tolist(),
            "hider_mean_return": round(hider_mean, 4),
            "seeker_mean_return": round(seeker_mean, 4),
            "first_visible_step": first_visible_step,
            "visible_fraction": round(float(visible_steps) / float(steps), 4),
        }
        episode_summaries.append(summary)
        print(
            "episode={episode} seed={seed} steps={steps} samples={samples} "
            "winner={winner} hider_mean={hider_mean_return} "
            "seeker_mean={seeker_mean_return} visible_fraction={visible_fraction} "
            "discard={discard_episode}".format(**summary)
        )

    arrays = stack_buffers(buffers)
    manifest["episodes_collected"] = len(episode_summaries)
    manifest["samples"] = int(len(arrays["role"]))
    manifest["discard_count"] = discard_count

    np.savez_compressed(
        args.out,
        manifest=np.array(manifest, dtype=object),
        episode_summaries=np.array(episode_summaries, dtype=object),
        episode_layouts=np.array(episode_layouts, dtype=object),
        **arrays
    )

    print("saved:", args.out)
    print("episodes_collected:", len(episode_summaries))
    print("samples:", manifest["samples"])
    print("discard_count:", discard_count)
    print("npz_keys:", sorted(list(arrays.keys()) + ["manifest", "episode_summaries", "episode_layouts"]))
    print("manifest_json:", json.dumps(make_serializable(manifest), sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
