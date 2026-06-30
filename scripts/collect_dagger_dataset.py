import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from collect_behavioral_dataset import (  # noqa: E402
    ROLE_HIDER,
    ROLE_SEEKER,
    append_episode_winner,
    agent_visible_flags,
    capture_layout,
    combine_team_actions,
    make_serializable,
    new_buffers,
    split_team_obs,
    stack_buffers,
    summarize_obs,
    winner_from_returns,
)
from evaluate_clone_vs_pretrained import ClonedBCPolicy, CloneRoleController  # noqa: E402
from ma_policy.load_policy import load_policy  # noqa: E402
from mujoco_worldgen.util.envs import load_env  # noqa: E402


def select_action(mode, expert_action, clone_action, expert_prob, rng):
    if mode == "expert":
        return expert_action, "expert"
    if mode == "clone":
        return clone_action, "clone"
    if rng.rand() < expert_prob:
        return expert_action, "expert"
    return clone_action, "clone"


def action_exact_by_agent(left, right):
    movement_equal = np.all(left["action_movement"] == right["action_movement"], axis=1)
    pull_equal = left["action_pull"].reshape(-1) == right["action_pull"].reshape(-1)
    glue_equal = left["action_glueall"].reshape(-1) == right["action_glueall"].reshape(-1)
    return movement_equal & pull_equal & glue_equal


def extend_dagger_buffers(
    buffers,
    obs,
    next_obs,
    expert_action,
    behavior_action,
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
        for key, value in expert_action.items():
            buffers["action"][key].append(np.asarray(value[agent_idx]).copy())
        for key, value in behavior_action.items():
            buffers["behavior_action"][key].append(np.asarray(value[agent_idx]).copy())


def new_dagger_buffers(obs_keys, action_keys):
    buffers = new_buffers(obs_keys, action_keys)
    buffers["behavior_action"] = {key: [] for key in action_keys}
    buffers["behavior_source"] = []
    return buffers


def append_behavior_sources(buffers, n_agents, hider_source, seeker_source, n_hiders):
    for agent_idx in range(n_agents):
        source = hider_source if agent_idx < n_hiders else seeker_source
        buffers["behavior_source"].append(0 if source == "expert" else 1)


def stack_dagger_buffers(buffers):
    arrays = stack_buffers(buffers)
    arrays["behavior_source"] = np.asarray(buffers["behavior_source"], dtype=np.int8)
    for key, values in buffers["behavior_action"].items():
        arrays["behavior_action_{}".format(key)] = np.asarray(values)
    return arrays


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
    parser.add_argument("--clone-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--n-hiders", type=int, default=0)
    parser.add_argument("--execute-hider", choices=["clone", "expert", "mix"], default="clone")
    parser.add_argument("--execute-seeker", choices=["clone", "expert", "mix"], default="clone")
    parser.add_argument("--mix-expert-prob", type=float, default=0.2)
    parser.add_argument("--out", default="/workspace/runs/dagger_hide_seek_full.npz")
    args = parser.parse_args()

    if not 0.0 <= args.mix_expert_prob <= 1.0:
        raise ValueError("--mix-expert-prob must be between 0 and 1")

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

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

    expert_hider = load_policy(args.hider_policy, env=env, scope="expert_hider", stochastic=False)
    expert_seeker = load_policy(args.seeker_policy, env=env, scope="expert_seeker", stochastic=False)
    clone_policy = ClonedBCPolicy(args.clone_dir)
    clone_hider = CloneRoleController(clone_policy, ROLE_HIDER)
    clone_seeker = CloneRoleController(clone_policy, ROLE_SEEKER)

    hider_obs, seeker_obs = split_team_obs(first_obs, n_hiders)
    expert_hider_action, _ = expert_hider.act(hider_obs)
    expert_seeker_action, _ = expert_seeker.act(seeker_obs)
    first_action = combine_team_actions(expert_hider_action, expert_seeker_action)
    buffers = new_dagger_buffers(sorted(first_obs.keys()), sorted(first_action.keys()))

    manifest = {
        "schema_version": "behavioral_dataset_v1",
        "collector": "collect_dagger_dataset.py",
        "created_unix": time.time(),
        "env": args.env,
        "hider_policy": args.hider_policy,
        "seeker_policy": args.seeker_policy,
        "clone_dir": args.clone_dir,
        "label_policy": "deterministic_pretrained_expert",
        "execute_hider": args.execute_hider,
        "execute_seeker": args.execute_seeker,
        "mix_expert_prob": args.mix_expert_prob,
        "episodes_requested": args.episodes,
        "max_steps_per_episode": args.steps,
        "seed": args.seed,
        "n_hiders": n_hiders,
        "role_encoding": {"hider": ROLE_HIDER, "seeker": ROLE_SEEKER},
        "behavior_source_encoding": {"expert": 0, "clone": 1},
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "observation_summary": summarize_obs(first_obs),
        "sample_axis": "one row per episode-step-agent",
        "notes": [
            "obs_* contains states visited by the behavior policy.",
            "action_* contains deterministic pretrained expert labels for those states.",
            "behavior_action_* contains the action actually executed in the environment.",
            "reward, done, next_obs_* and episode_winner come from the behavior-policy rollout.",
        ],
    }

    # The first expert call above was only for schema discovery; reset before real collection.
    expert_hider.reset()
    expert_seeker.reset()
    clone_hider.reset()
    clone_seeker.reset()

    episode_summaries = []
    episode_layouts = []
    discard_count = 0

    for episode_idx in range(args.episodes):
        seed = args.seed + episode_idx
        env.seed(seed)
        obs = env.reset()
        expert_hider.reset()
        expert_seeker.reset()
        clone_hider.reset()
        clone_seeker.reset()

        episode_layouts.append(capture_layout(env))
        start_count = len(buffers["episode"])
        total_reward = None
        visible_steps = 0
        first_visible_step = None
        done = False
        info = {}
        hider_expert_matches = []
        seeker_expert_matches = []
        hider_behavior_sources = []
        seeker_behavior_sources = []

        for step_idx in range(args.steps):
            n_agents = len(next(iter(obs.values())))
            visible_flags = agent_visible_flags(obs, n_hiders, n_agents)
            if np.any(visible_flags):
                visible_steps += 1
                if first_visible_step is None:
                    first_visible_step = step_idx

            hider_obs, seeker_obs = split_team_obs(obs, n_hiders)
            expert_hider_action, _ = expert_hider.act(hider_obs)
            expert_seeker_action, _ = expert_seeker.act(seeker_obs)
            clone_hider_action, _ = clone_hider.act(hider_obs)
            clone_seeker_action, _ = clone_seeker.act(seeker_obs)

            behavior_hider_action, hider_source = select_action(
                args.execute_hider,
                expert_hider_action,
                clone_hider_action,
                args.mix_expert_prob,
                rng,
            )
            behavior_seeker_action, seeker_source = select_action(
                args.execute_seeker,
                expert_seeker_action,
                clone_seeker_action,
                args.mix_expert_prob,
                rng,
            )

            expert_action = combine_team_actions(expert_hider_action, expert_seeker_action)
            behavior_action = combine_team_actions(behavior_hider_action, behavior_seeker_action)
            hider_expert_matches.extend(action_exact_by_agent(behavior_hider_action, expert_hider_action).tolist())
            seeker_expert_matches.extend(action_exact_by_agent(behavior_seeker_action, expert_seeker_action).tolist())
            hider_behavior_sources.append(hider_source)
            seeker_behavior_sources.append(seeker_source)

            next_obs, reward, done, info = env.step(behavior_action)
            reward = np.asarray(reward)
            total_reward = reward if total_reward is None else total_reward + reward

            extend_dagger_buffers(
                buffers=buffers,
                obs=obs,
                next_obs=next_obs,
                expert_action=expert_action,
                behavior_action=behavior_action,
                reward=reward,
                done=done,
                visible_flags=visible_flags,
                episode_idx=episode_idx,
                seed=seed,
                step_idx=step_idx,
                n_hiders=n_hiders,
            )
            append_behavior_sources(buffers, n_agents, hider_source, seeker_source, n_hiders)

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
            "hider_behavior_expert_action_exact_rate": round(float(np.mean(hider_expert_matches)), 4),
            "seeker_behavior_expert_action_exact_rate": round(float(np.mean(seeker_expert_matches)), 4),
            "hider_expert_behavior_fraction": round(float(hider_behavior_sources.count("expert")) / float(steps), 4),
            "seeker_expert_behavior_fraction": round(float(seeker_behavior_sources.count("expert")) / float(steps), 4),
        }
        episode_summaries.append(summary)
        print(
            "episode={episode} seed={seed} steps={steps} samples={samples} "
            "winner={winner} hider_mean={hider_mean_return} seeker_mean={seeker_mean_return} "
            "visible_fraction={visible_fraction} hider_match={hider_behavior_expert_action_exact_rate} "
            "seeker_match={seeker_behavior_expert_action_exact_rate} discard={discard_episode}".format(**summary)
        )

    arrays = stack_dagger_buffers(buffers)
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
