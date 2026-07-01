import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from ma_policy.load_policy import load_policy
from mujoco_worldgen.util.envs import load_env


ROLE_HIDER = 0
ROLE_SEEKER = 1


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


def checkpoint_prefix(clone_dir):
    checkpoint_file = os.path.join(clone_dir, "checkpoint")
    if not os.path.exists(checkpoint_file):
        return os.path.join(clone_dir, "model.ckpt")
    state = tf.train.get_checkpoint_state(clone_dir)
    if state is None or not state.model_checkpoint_path:
        return os.path.join(clone_dir, "model.ckpt")
    path = state.model_checkpoint_path
    return path if os.path.isabs(path) else os.path.join(clone_dir, path)


class ClonedBCPolicy(object):
    def __init__(self, clone_dir, graph=None, session=None):
        self.clone_dir = clone_dir
        self.graph = graph or tf.get_default_graph()
        self.session = session or tf.get_default_session()
        if self.session is None:
            raise ValueError("ClonedBCPolicy requires an active TensorFlow session")

        preprocessing_path = os.path.join(clone_dir, "preprocessing.npz")
        if not os.path.exists(preprocessing_path):
            raise ValueError("Missing preprocessing artifact: {}".format(preprocessing_path))

        with self.graph.as_default():
            preprocessing = np.load(preprocessing_path, allow_pickle=True)
            self.model_type = str(preprocessing["model_type"].item()) if "model_type" in preprocessing.files else "flat_bc_v1"
            self.preprocessing = preprocessing

            ckpt = checkpoint_prefix(clone_dir)
            meta_path = ckpt + ".meta"
            if not os.path.exists(meta_path):
                raise ValueError("Missing checkpoint meta graph: {}".format(meta_path))

            self.saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
            self.saver.restore(self.session, ckpt)
            self.pred_tensors = {
                "movement_0": self.graph.get_tensor_by_name("pred_movement_0:0"),
                "movement_1": self.graph.get_tensor_by_name("pred_movement_1:0"),
                "movement_2": self.graph.get_tensor_by_name("pred_movement_2:0"),
                "pull": self.graph.get_tensor_by_name("pred_pull:0"),
                "glueall": self.graph.get_tensor_by_name("pred_glueall:0"),
            }
            if self.model_type == "structured_bc_v1":
                self.obs_keys = [str(value) for value in preprocessing["obs_keys"].tolist()]
                self.float_obs_keys = [str(value) for value in preprocessing["float_obs_keys"].tolist()]
                self.input_tensors = {
                    key: self.graph.get_tensor_by_name("input_{}:0".format(key))
                    for key in self.obs_keys
                }
                self.role_ph = self.graph.get_tensor_by_name("role:0")
            else:
                self.feature_mean = preprocessing["feature_mean"].astype(np.float32)
                self.feature_std = preprocessing["feature_std"].astype(np.float32)
                self.obs_keys = [str(value) for value in preprocessing["obs_keys"].tolist()]
                self.include_role = bool(preprocessing["include_role"].item())
                self.features_ph = self.graph.get_tensor_by_name("features:0")

    def reset(self):
        return None

    def _flat_features(self, observation, role_id):
        rows = []
        for obs_key in self.obs_keys:
            runtime_key = obs_key[4:] if obs_key.startswith("obs_") else obs_key
            if runtime_key not in observation:
                raise ValueError("Observation key not present at rollout time: {}".format(runtime_key))
            value = np.asarray(observation[runtime_key])
            rows.append(value.reshape((value.shape[0], -1)).astype(np.float32))

        if self.include_role:
            n_agents = rows[0].shape[0]
            role_onehot = np.zeros((n_agents, 2), dtype=np.float32)
            role_onehot[:, role_id] = 1.0
            rows.append(role_onehot)

        features = np.concatenate(rows, axis=1)
        return (features - self.feature_mean) / self.feature_std

    def _structured_feed(self, observation, role_id):
        n_agents = None
        feed = {}
        for key in self.obs_keys:
            if key not in observation:
                raise ValueError("Observation key not present at rollout time: {}".format(key))
            value = np.asarray(observation[key]).astype(np.float32)
            if n_agents is None:
                n_agents = value.shape[0]
            if key in self.float_obs_keys:
                mean = self.preprocessing["mean_{}".format(key)].astype(np.float32)
                std = self.preprocessing["std_{}".format(key)].astype(np.float32)
                value = (value - mean) / std
            feed[self.input_tensors[key]] = value
        feed[self.role_ph] = np.full((n_agents,), role_id, dtype=np.int32)
        return feed

    def act_for_role(self, observation, role_id):
        if self.model_type == "structured_bc_v1":
            feed = self._structured_feed(observation, role_id)
        else:
            features = self._flat_features(observation, role_id)
            feed = {self.features_ph: features}
        outputs = self.session.run(self.pred_tensors, feed_dict=feed)
        movement = np.stack([
            outputs["movement_0"],
            outputs["movement_1"],
            outputs["movement_2"],
        ], axis=1).astype(np.int64)
        return {
            "action_movement": movement,
            "action_pull": outputs["pull"].astype(np.int64),
            "action_glueall": outputs["glueall"].astype(np.int64),
        }, {}


class CloneRoleController(object):
    def __init__(self, clone_policy, role_id):
        self.clone_policy = clone_policy
        self.role_id = role_id

    def reset(self):
        return self.clone_policy.reset()

    def act(self, observation):
        return self.clone_policy.act_for_role(observation, self.role_id)


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
    hider_mean = float(np.mean(total_reward[:n_hiders]))
    seeker_mean = float(np.mean(total_reward[n_hiders:]))

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
    }


def summarize_case(case_name, episodes):
    winners = [episode["winner"] for episode in episodes]
    return {
        "case": case_name,
        "episodes": len(episodes),
        "hider_wins": winners.count("hiders"),
        "seeker_wins": winners.count("seekers"),
        "ties": winners.count("tie"),
        "hider_win_rate": round(float(winners.count("hiders")) / max(len(winners), 1), 4),
        "seeker_win_rate": round(float(winners.count("seekers")) / max(len(winners), 1), 4),
        "tie_rate": round(float(winners.count("tie")) / max(len(winners), 1), 4),
        "mean_hider_return": round(float(np.mean([episode["hider_mean_return"] for episode in episodes])), 4),
        "mean_seeker_return": round(float(np.mean([episode["seeker_mean_return"] for episode in episodes])), 4),
        "mean_visible_fraction": round(float(np.mean([episode["visible_fraction"] for episode in episodes])), 4),
        "discard_count": int(sum(1 for episode in episodes if episode["discard_episode"])),
        "episodes_detail": episodes,
    }


def write_report(path, summary):
    lines = []
    lines.append("Clone Vs Pretrained Closed-Loop Evaluation")
    lines.append("episodes_per_case: {}".format(summary["episodes_per_case"]))
    lines.append("steps: {}".format(summary["steps"]))
    lines.append("seed_start: {}".format(summary["seed"]))
    lines.append("hider_clone_dir: {}".format(summary["hider_clone_dir"]))
    lines.append("seeker_clone_dir: {}".format(summary["seeker_clone_dir"]))
    lines.append("")
    lines.append("{:<28} {:>4} {:>4} {:>4} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
        "case", "S", "H", "T", "S_win", "H_win", "T_rate", "visible", "S_return", "H_return"
    ))
    lines.append("-" * 112)
    for case in summary["cases"]:
        lines.append("{:<28} {:>4} {:>4} {:>4} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.3f} {:>10.3f}".format(
            case["case"],
            case["seeker_wins"],
            case["hider_wins"],
            case["ties"],
            case["seeker_win_rate"],
            case["hider_win_rate"],
            case["tie_rate"],
            case["mean_visible_fraction"],
            case["mean_seeker_return"],
            case["mean_hider_return"],
        ))

    with open(path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


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
    parser.add_argument("--clone-dir", default="/workspace/runs/bc_baseline_d_ramp_vs_c_ramps_det_20ep")
    parser.add_argument("--hider-clone-dir", default=None)
    parser.add_argument("--seeker-clone-dir", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--n-hiders", type=int, default=0)
    parser.add_argument("--pretrained-stochastic", action="store_true")
    parser.add_argument("--out", default="/workspace/runs/clone_vs_pretrained/summary.json")
    parser.add_argument("--report-out", default="/workspace/runs/clone_vs_pretrained/report.txt")
    args = parser.parse_args()

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
    env.reset()

    n_hiders = args.n_hiders or int(env.metadata.get("n_hiders", 0))
    if n_hiders <= 0:
        raise ValueError("Could not determine n_hiders; pass --n-hiders explicitly")

    pretrained_hider = load_policy(
        args.hider_policy,
        env=env,
        scope="pretrained_hider",
        stochastic=args.pretrained_stochastic,
    )
    pretrained_seeker = load_policy(
        args.seeker_policy,
        env=env,
        scope="pretrained_seeker",
        stochastic=args.pretrained_stochastic,
    )
    hider_clone_dir = args.hider_clone_dir or args.clone_dir
    seeker_clone_dir = args.seeker_clone_dir or args.clone_dir

    if hider_clone_dir == seeker_clone_dir:
        clone_policy = ClonedBCPolicy(hider_clone_dir)
        cloned_hider = CloneRoleController(clone_policy, ROLE_HIDER)
        cloned_seeker = CloneRoleController(clone_policy, ROLE_SEEKER)
    else:
        hider_graph = tf.Graph()
        hider_sess = tf.Session(graph=hider_graph, config=tf_config)
        seeker_graph = tf.Graph()
        seeker_sess = tf.Session(graph=seeker_graph, config=tf_config)
        hider_clone_policy = ClonedBCPolicy(hider_clone_dir, graph=hider_graph, session=hider_sess)
        seeker_clone_policy = ClonedBCPolicy(seeker_clone_dir, graph=seeker_graph, session=seeker_sess)
        cloned_hider = CloneRoleController(hider_clone_policy, ROLE_HIDER)
        cloned_seeker = CloneRoleController(seeker_clone_policy, ROLE_SEEKER)

    case_specs = [
        ("pretrained_vs_pretrained", pretrained_hider, pretrained_seeker),
        ("clone_hider_vs_pretrained", cloned_hider, pretrained_seeker),
        ("pretrained_vs_clone_seeker", pretrained_hider, cloned_seeker),
        ("clone_vs_clone", cloned_hider, cloned_seeker),
    ]

    cases = []
    for case_name, hider_controller, seeker_controller in case_specs:
        print("=== case: {} ===".format(case_name))
        episodes = []
        for episode_idx in range(args.episodes):
            result = evaluate_episode(
                env=env,
                hider_policy=hider_controller,
                seeker_policy=seeker_controller,
                seed=args.seed + episode_idx,
                max_steps=args.steps,
                n_hiders=n_hiders,
            )
            episodes.append(result)
            print(
                "case={case} episode={idx} seed={seed} winner={winner} "
                "hider_mean={hider_mean_return} seeker_mean={seeker_mean_return} "
                "visible_fraction={visible_fraction} discard={discard_episode}".format(
                    case=case_name,
                    idx=episode_idx,
                    **result
                )
            )
        case_summary = summarize_case(case_name, episodes)
        cases.append(case_summary)
        print(
            "case_summary case={case} seeker_win_rate={seeker_win_rate} "
            "hider_win_rate={hider_win_rate} visible={mean_visible_fraction} "
            "discard_count={discard_count}".format(**case_summary)
        )

    summary = {
        "created_unix": time.time(),
        "env": args.env,
        "hider_policy": args.hider_policy,
        "seeker_policy": args.seeker_policy,
        "clone_dir": args.clone_dir if hider_clone_dir == seeker_clone_dir else None,
        "fallback_clone_dir": args.clone_dir,
        "hider_clone_dir": hider_clone_dir,
        "seeker_clone_dir": seeker_clone_dir,
        "pretrained_stochastic": args.pretrained_stochastic,
        "episodes_per_case": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "n_hiders": n_hiders,
        "cases": cases,
    }

    for path in [args.out, args.report_out]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as output_file:
        json.dump(make_serializable(summary), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    write_report(args.report_out, summary)

    print("saved_summary:", args.out)
    print("saved_report:", args.report_out)


if __name__ == "__main__":
    main()
