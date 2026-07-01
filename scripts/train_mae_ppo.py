import argparse
import csv
import json
import os
import time

import numpy as np
import tensorflow as tf

from mujoco_worldgen.util.envs import load_env


MOVEMENT_CLASSES = 11
BINARY_CLASSES = 2
ACTION_KEYS = ["movement_0", "movement_1", "movement_2", "pull", "glueall"]


def parse_hidden_sizes(value):
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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


class RunningMeanStd(object):
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = float(x.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x, clip=10.0):
        normalized = (np.asarray(x, dtype=np.float32) - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(normalized, -clip, clip).astype(np.float32)


def flatten_obs(obs, obs_keys):
    pieces = []
    for key in obs_keys:
        value = np.asarray(obs[key], dtype=np.float32)
        pieces.append(value.reshape((value.shape[0], -1)))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def critic_features(actor_features):
    global_features = actor_features.reshape(1, -1)
    global_repeated = np.repeat(global_features, actor_features.shape[0], axis=0)
    return np.concatenate([actor_features, global_repeated], axis=1).astype(np.float32)


def roles_for_env(n_agents, n_hiders):
    roles = np.ones(n_agents, dtype=np.int32)
    roles[:n_hiders] = 0
    return roles


def actions_to_dict(actions):
    movement = np.stack(
        [actions["movement_0"], actions["movement_1"], actions["movement_2"]],
        axis=1,
    ).astype(np.int32)
    return {
        "action_movement": movement,
        "action_pull": actions["pull"].astype(np.int32),
        "action_glueall": actions["glueall"].astype(np.int32),
    }


def split_action_dict(action_dict):
    movement = np.asarray(action_dict["action_movement"], dtype=np.int32)
    return {
        "movement_0": movement[:, 0],
        "movement_1": movement[:, 1],
        "movement_2": movement[:, 2],
        "pull": np.asarray(action_dict["action_pull"], dtype=np.int32),
        "glueall": np.asarray(action_dict["action_glueall"], dtype=np.int32),
    }


def dense_stack(x, sizes, scope):
    h = x
    with tf.variable_scope(scope):
        for idx, size in enumerate(sizes):
            h = tf.layers.dense(h, size, activation=tf.nn.tanh, name="dense_{}".format(idx))
    return h


def policy_value_scope(actor_x, critic_x, hidden_sizes, scope):
    with tf.variable_scope(scope):
        actor_h = dense_stack(actor_x, hidden_sizes, "actor")
        critic_h = dense_stack(critic_x, hidden_sizes, "critic")
        logits = {
            "movement_0": tf.layers.dense(actor_h, MOVEMENT_CLASSES, name="logits_movement_0"),
            "movement_1": tf.layers.dense(actor_h, MOVEMENT_CLASSES, name="logits_movement_1"),
            "movement_2": tf.layers.dense(actor_h, MOVEMENT_CLASSES, name="logits_movement_2"),
            "pull": tf.layers.dense(actor_h, BINARY_CLASSES, name="logits_pull"),
            "glueall": tf.layers.dense(actor_h, BINARY_CLASSES, name="logits_glueall"),
        }
        value = tf.squeeze(tf.layers.dense(critic_h, 1, name="value"), axis=1)
    return logits, value


def categorical_log_prob(logits, action):
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action, logits=logits)


def categorical_entropy(logits):
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(probs * log_probs, axis=1)


def select_by_role(hider_value, seeker_value, role):
    selector = tf.cast(tf.equal(role, 0), tf.float32)
    while len(selector.shape) < len(hider_value.shape):
        selector = tf.expand_dims(selector, -1)
    return selector * hider_value + (1.0 - selector) * seeker_value


def build_model(actor_dim, critic_dim, hidden_sizes, learning_rate, clip_range, value_coef, entropy_coef, max_grad_norm, reset_graph=True):
    if reset_graph:
        tf.reset_default_graph()
    actor_x = tf.placeholder(tf.float32, shape=[None, actor_dim], name="actor_features")
    critic_x = tf.placeholder(tf.float32, shape=[None, critic_dim], name="critic_features")
    role = tf.placeholder(tf.int32, shape=[None], name="role")
    old_logp = tf.placeholder(tf.float32, shape=[None], name="old_logp")
    old_value = tf.placeholder(tf.float32, shape=[None], name="old_value")
    advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")
    return_target = tf.placeholder(tf.float32, shape=[None], name="return")
    actions = {
        "movement_0": tf.placeholder(tf.int32, shape=[None], name="action_movement_0"),
        "movement_1": tf.placeholder(tf.int32, shape=[None], name="action_movement_1"),
        "movement_2": tf.placeholder(tf.int32, shape=[None], name="action_movement_2"),
        "pull": tf.placeholder(tf.int32, shape=[None], name="action_pull"),
        "glueall": tf.placeholder(tf.int32, shape=[None], name="action_glueall"),
    }

    hider_logits, hider_value = policy_value_scope(actor_x, critic_x, hidden_sizes, "hider")
    seeker_logits, seeker_value = policy_value_scope(actor_x, critic_x, hidden_sizes, "seeker")

    logits = {
        key: select_by_role(hider_logits[key], seeker_logits[key], role)
        for key in ACTION_KEYS
    }
    value = select_by_role(hider_value, seeker_value, role)

    sampled_actions = {
        key: tf.cast(tf.squeeze(tf.multinomial(logits[key], num_samples=1), axis=1), tf.int32)
        for key in ACTION_KEYS
    }
    greedy_actions = {
        key: tf.argmax(logits[key], axis=1, output_type=tf.int32)
        for key in ACTION_KEYS
    }

    logp_parts = [categorical_log_prob(logits[key], actions[key]) for key in ACTION_KEYS]
    new_logp = tf.add_n(logp_parts, name="logp")
    entropy = tf.add_n([categorical_entropy(logits[key]) for key in ACTION_KEYS], name="entropy")

    ratio = tf.exp(new_logp - old_logp)
    unclipped_policy_loss = ratio * advantage
    clipped_policy_loss = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
    policy_loss = -tf.reduce_mean(tf.minimum(unclipped_policy_loss, clipped_policy_loss), name="policy_loss")

    value_pred_clipped = old_value + tf.clip_by_value(value - old_value, -clip_range, clip_range)
    value_losses = tf.square(value - return_target)
    value_losses_clipped = tf.square(value_pred_clipped - return_target)
    value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_losses, value_losses_clipped), name="value_loss")

    entropy_mean = tf.reduce_mean(entropy, name="entropy_mean")
    approx_kl = tf.reduce_mean(old_logp - new_logp, name="approx_kl")
    clip_fraction = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_range), tf.float32), name="clip_fraction")
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    params = tf.trainable_variables()
    gradients = tf.gradients(total_loss, params)
    gradients, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
    train_op = optimizer.apply_gradients(zip(gradients, params))

    return {
        "actor_x": actor_x,
        "critic_x": critic_x,
        "role": role,
        "logits": logits,
        "actions": actions,
        "old_logp": old_logp,
        "old_value": old_value,
        "advantage": advantage,
        "return": return_target,
        "sampled_actions": sampled_actions,
        "greedy_actions": greedy_actions,
        "value": value,
        "new_logp": new_logp,
        "train_op": train_op,
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "grad_norm": grad_norm,
    }


def feed_actions(model, action_arrays):
    return {
        model["actions"][key]: action_arrays[key]
        for key in ACTION_KEYS
    }


def act(sess, model, actor_batch, critic_batch, role_batch, actor_rms, critic_rms, deterministic=False):
    actor_norm = actor_rms.normalize(actor_batch)
    critic_norm = critic_rms.normalize(critic_batch)
    action_fetch = model["greedy_actions"] if deterministic else model["sampled_actions"]
    sampled = sess.run(
        action_fetch,
        feed_dict={
            model["actor_x"]: actor_norm,
            model["critic_x"]: critic_norm,
            model["role"]: role_batch,
        },
    )
    feed = {
        model["actor_x"]: actor_norm,
        model["critic_x"]: critic_norm,
        model["role"]: role_batch,
    }
    feed.update(feed_actions(model, sampled))
    logp_value = sess.run(
        {"logp": model["new_logp"], "value": model["value"]},
        feed_dict=feed,
    )
    return sampled, logp_value["logp"], logp_value["value"]


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    rollout_steps, n_agents = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros((n_agents,), dtype=np.float32)
    for step in reversed(range(rollout_steps)):
        if step == rollout_steps - 1:
            next_values = last_values
            next_nonterminal = 1.0 - dones[step]
        else:
            next_values = values[step + 1]
            next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def minibatches(n_samples, batch_size, rng):
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        yield indices[start:start + batch_size]


def summarize_winner(total_reward, n_hiders):
    total_reward = np.asarray(total_reward, dtype=np.float32)
    hider_mean = float(np.mean(total_reward[:n_hiders]))
    seeker_mean = float(np.mean(total_reward[n_hiders:]))
    if seeker_mean > hider_mean:
        return "seekers"
    if hider_mean > seeker_mean:
        return "hiders"
    return "tie"


def evaluate_policy(sess, env, model, actor_rms, critic_rms, obs_keys, roles, n_hiders, episodes, max_steps, seed):
    summaries = []
    for episode_idx in range(episodes):
        env.seed(seed + episode_idx)
        obs = env.reset()
        total_reward = None
        done = False
        info = {}
        visible_steps = 0
        for step_idx in range(max_steps):
            actor_batch = flatten_obs(obs, obs_keys)
            critic_batch = critic_features(actor_batch)
            if "mask_aa_obs" in obs:
                visible = np.any(np.asarray(obs["mask_aa_obs"])[n_hiders:, :n_hiders])
                visible_steps += int(visible)
            actions, _, _ = act(
                sess,
                model,
                actor_batch,
                critic_batch,
                roles,
                actor_rms,
                critic_rms,
                deterministic=True,
            )
            obs, reward, done, info = env.step(actions_to_dict(actions))
            total_reward = reward if total_reward is None else total_reward + reward
            if done or info.get("discard_episode"):
                break
        steps = step_idx + 1
        total_reward = np.asarray(total_reward, dtype=np.float32)
        summaries.append({
            "episode": episode_idx,
            "seed": seed + episode_idx,
            "steps": steps,
            "done": bool(done),
            "discard_episode": bool(info.get("discard_episode")),
            "total_reward": total_reward.round(4).tolist(),
            "hider_mean_return": round(float(np.mean(total_reward[:n_hiders])), 4),
            "seeker_mean_return": round(float(np.mean(total_reward[n_hiders:])), 4),
            "winner": summarize_winner(total_reward, n_hiders),
            "visible_fraction": round(float(visible_steps) / float(max(steps, 1)), 4),
        })
    return summaries


def write_report(path, summary):
    lines = [
        "MAE PPO-GAE Training Report",
        "env: {}".format(summary["env"]),
        "updates: {}".format(summary["updates_completed"]),
        "rollout_steps: {}".format(summary["rollout_steps"]),
        "checkpoint: {}".format(summary["checkpoint_path"]),
        "hider_variables: {}".format(summary["hider_variables_path"]),
        "seeker_variables: {}".format(summary["seeker_variables_path"]),
        "",
        "Final training row:",
        json.dumps(summary["last_progress"], indent=2, sort_keys=True),
    ]
    if summary.get("evaluation"):
        winners = [episode["winner"] for episode in summary["evaluation"]]
        lines.extend([
            "",
            "Final deterministic evaluation:",
            "episodes: {}".format(len(winners)),
            "hider_wins: {}".format(winners.count("hiders")),
            "seeker_wins: {}".format(winners.count("seekers")),
            "ties: {}".format(winners.count("tie")),
            "mean_visible_fraction: {:.4f}".format(float(np.mean([episode["visible_fraction"] for episode in summary["evaluation"]]))),
        ])
    with open(path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def append_progress_row(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    fieldnames = sorted(row.keys())
    with open(path, "a", newline="") as progress_file:
        writer = csv.DictWriter(progress_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet",
    )
    parser.add_argument("--out-dir", default="/workspace/runs/mae_ppo_quadrant_v1")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    env = load_env(
        args.env,
        core_dir="/workspace/multi-agent-emergence-environments",
        envs_dir="mae_envs/envs",
        xmls_dir="xmls",
    )
    env.seed(args.seed)
    obs = env.reset()
    obs_keys = sorted(obs.keys())
    actor_example = flatten_obs(obs, obs_keys)
    critic_example = critic_features(actor_example)
    n_agents = actor_example.shape[0]
    n_hiders = int(env.metadata.get("n_hiders", 0))
    if n_hiders <= 0:
        raise ValueError("Could not determine n_hiders from environment metadata")
    roles = roles_for_env(n_agents, n_hiders)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    actor_rms = RunningMeanStd(shape=(actor_example.shape[1],))
    critic_rms = RunningMeanStd(shape=(critic_example.shape[1],))

    model = build_model(
        actor_dim=actor_example.shape[1],
        critic_dim=critic_example.shape[1],
        hidden_sizes=hidden_sizes,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )
    saver = tf.train.Saver(max_to_keep=5)
    hider_variables_path = os.path.join(args.out_dir, "hider_variables.txt")
    seeker_variables_path = os.path.join(args.out_dir, "seeker_variables.txt")
    with open(hider_variables_path, "w") as output_file:
        output_file.write("\n".join(sorted([var.name for var in tf.trainable_variables(scope="hider")])))
        output_file.write("\n")
    with open(seeker_variables_path, "w") as output_file:
        output_file.write("\n".join(sorted([var.name for var in tf.trainable_variables(scope="seeker")])))
        output_file.write("\n")

    progress_path = os.path.join(args.out_dir, "progress.csv")
    if os.path.exists(progress_path):
        os.remove(progress_path)
    progress_rows = []
    episode_returns = []
    episode_lengths = []
    episode_winners = []
    current_return = np.zeros((n_agents,), dtype=np.float32)
    current_length = 0
    last_progress = {}
    checkpoint_path = ""
    evaluation = []

    max_episode_steps = args.max_episode_steps or int(getattr(env.unwrapped, "horizon", args.rollout_steps))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for update in range(1, args.updates + 1):
            actor_buf = []
            critic_buf = []
            role_buf = []
            action_buf = {key: [] for key in ACTION_KEYS}
            logp_buf = []
            value_buf = []
            reward_buf = []
            done_buf = []

            for _ in range(args.rollout_steps):
                actor_batch = flatten_obs(obs, obs_keys)
                critic_batch = critic_features(actor_batch)
                actor_rms.update(actor_batch)
                critic_rms.update(critic_batch)

                actions, logp, values = act(
                    sess,
                    model,
                    actor_batch,
                    critic_batch,
                    roles,
                    actor_rms,
                    critic_rms,
                    deterministic=False,
                )
                next_obs, reward, done, info = env.step(actions_to_dict(actions))
                reward = np.asarray(reward, dtype=np.float32)
                done_flag = bool(done or info.get("discard_episode"))

                actor_buf.append(actor_batch)
                critic_buf.append(critic_batch)
                role_buf.append(roles.copy())
                for key in ACTION_KEYS:
                    action_buf[key].append(actions[key])
                logp_buf.append(logp)
                value_buf.append(values)
                reward_buf.append(reward)
                done_buf.append(np.ones((n_agents,), dtype=np.float32) * float(done_flag))

                current_return += reward
                current_length += 1
                obs = next_obs

                if done_flag or current_length >= max_episode_steps:
                    episode_returns.append(current_return.copy())
                    episode_lengths.append(current_length)
                    episode_winners.append(summarize_winner(current_return, n_hiders))
                    obs = env.reset()
                    current_return = np.zeros((n_agents,), dtype=np.float32)
                    current_length = 0

            last_actor = flatten_obs(obs, obs_keys)
            last_critic = critic_features(last_actor)
            _, _, last_values = act(
                sess,
                model,
                last_actor,
                last_critic,
                roles,
                actor_rms,
                critic_rms,
                deterministic=False,
            )
            if len(done_buf) and np.all(done_buf[-1] > 0.0):
                last_values = np.zeros_like(last_values)

            actor_arr = np.asarray(actor_buf, dtype=np.float32)
            critic_arr = np.asarray(critic_buf, dtype=np.float32)
            role_arr = np.asarray(role_buf, dtype=np.int32)
            logp_arr = np.asarray(logp_buf, dtype=np.float32)
            value_arr = np.asarray(value_buf, dtype=np.float32)
            reward_arr = np.asarray(reward_buf, dtype=np.float32)
            done_arr = np.asarray(done_buf, dtype=np.float32)
            action_arr = {key: np.asarray(value, dtype=np.int32) for key, value in action_buf.items()}

            advantages, returns = compute_gae(
                rewards=reward_arr,
                values=value_arr,
                dones=done_arr,
                last_values=last_values,
                gamma=args.gamma,
                lam=args.gae_lambda,
            )
            flat_actor = actor_arr.reshape((-1, actor_arr.shape[-1]))
            flat_critic = critic_arr.reshape((-1, critic_arr.shape[-1]))
            flat_roles = role_arr.reshape((-1,))
            flat_logp = logp_arr.reshape((-1,))
            flat_values = value_arr.reshape((-1,))
            flat_adv = advantages.reshape((-1,))
            flat_returns = returns.reshape((-1,))
            flat_actions = {
                key: value.reshape((-1,))
                for key, value in action_arr.items()
            }

            adv_mean = np.mean(flat_adv)
            adv_std = np.std(flat_adv)
            flat_adv_norm = ((flat_adv - adv_mean) / max(adv_std, 1e-8)).astype(np.float32)

            train_metrics = []
            for _ in range(args.ppo_epochs):
                for batch_idx in minibatches(len(flat_actor), args.batch_size, rng):
                    feed = {
                        model["actor_x"]: actor_rms.normalize(flat_actor[batch_idx]),
                        model["critic_x"]: critic_rms.normalize(flat_critic[batch_idx]),
                        model["role"]: flat_roles[batch_idx],
                        model["old_logp"]: flat_logp[batch_idx],
                        model["old_value"]: flat_values[batch_idx],
                        model["advantage"]: flat_adv_norm[batch_idx],
                        model["return"]: flat_returns[batch_idx],
                    }
                    feed.update(feed_actions(model, {key: value[batch_idx] for key, value in flat_actions.items()}))
                    result = sess.run(
                        {
                            "train": model["train_op"],
                            "loss": model["loss"],
                            "policy_loss": model["policy_loss"],
                            "value_loss": model["value_loss"],
                            "entropy": model["entropy"],
                            "approx_kl": model["approx_kl"],
                            "clip_fraction": model["clip_fraction"],
                            "grad_norm": model["grad_norm"],
                        },
                        feed_dict=feed,
                    )
                    result.pop("train")
                    train_metrics.append(result)

            recent_returns = episode_returns[-20:]
            recent_winners = episode_winners[-20:]
            mean_team_returns = np.mean(recent_returns, axis=0) if recent_returns else np.zeros((n_agents,), dtype=np.float32)
            metric_mean = {
                key: round(float(np.mean([row[key] for row in train_metrics])), 6)
                for key in train_metrics[0]
            }
            last_progress = {
                "update": update,
                "samples": int(len(flat_actor)),
                "episodes_completed": int(len(episode_returns)),
                "recent_hider_wins": int(recent_winners.count("hiders")),
                "recent_seeker_wins": int(recent_winners.count("seekers")),
                "recent_ties": int(recent_winners.count("tie")),
                "recent_mean_hider_return": round(float(np.mean(mean_team_returns[:n_hiders])), 4),
                "recent_mean_seeker_return": round(float(np.mean(mean_team_returns[n_hiders:])), 4),
                "recent_mean_episode_length": round(float(np.mean(episode_lengths[-20:])), 4) if episode_lengths else 0.0,
                "advantage_mean": round(float(adv_mean), 6),
                "advantage_std": round(float(adv_std), 6),
            }
            last_progress.update(metric_mean)
            progress_rows.append(last_progress)
            append_progress_row(progress_path, last_progress)

            print(
                "update={update} episodes={episodes_completed} hider_wins={recent_hider_wins} "
                "seeker_wins={recent_seeker_wins} hider_return={recent_mean_hider_return:.3f} "
                "seeker_return={recent_mean_seeker_return:.3f} loss={loss:.4f} "
                "kl={approx_kl:.5f} entropy={entropy:.4f}".format(**last_progress)
                , flush=True
            )

            if update % args.eval_every == 0 or update == args.updates:
                evaluation = evaluate_policy(
                    sess=sess,
                    env=env,
                    model=model,
                    actor_rms=actor_rms,
                    critic_rms=critic_rms,
                    obs_keys=obs_keys,
                    roles=roles,
                    n_hiders=n_hiders,
                    episodes=args.eval_episodes,
                    max_steps=max_episode_steps,
                    seed=args.seed + 100000 + update * 100,
                )
                winners = [episode["winner"] for episode in evaluation]
                print(
                    "evaluation update={} hider_wins={} seeker_wins={} ties={} mean_visible={:.4f}".format(
                        update,
                        winners.count("hiders"),
                        winners.count("seekers"),
                        winners.count("tie"),
                        float(np.mean([episode["visible_fraction"] for episode in evaluation])),
                    )
                    , flush=True
                )
                obs = env.reset()

            if update % args.save_every == 0 or update == args.updates:
                checkpoint_path = saver.save(sess, os.path.join(args.out_dir, "model.ckpt"), global_step=update)

        if not checkpoint_path:
            checkpoint_path = saver.save(sess, os.path.join(args.out_dir, "model.ckpt"), global_step=args.updates)

    if progress_rows:
        fieldnames = sorted(progress_rows[0].keys())
        with open(progress_path, "w", newline="") as progress_file:
            writer = csv.DictWriter(progress_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in progress_rows:
                writer.writerow(row)

    np.savez_compressed(
        os.path.join(args.out_dir, "normalization.npz"),
        obs_keys=np.asarray(obs_keys, dtype=object),
        actor_mean=actor_rms.mean.astype(np.float32),
        actor_var=actor_rms.var.astype(np.float32),
        actor_count=np.asarray(actor_rms.count, dtype=np.float32),
        critic_mean=critic_rms.mean.astype(np.float32),
        critic_var=critic_rms.var.astype(np.float32),
        critic_count=np.asarray(critic_rms.count, dtype=np.float32),
        roles=roles,
        n_hiders=np.asarray(n_hiders, dtype=np.int32),
        n_agents=np.asarray(n_agents, dtype=np.int32),
    )

    summary = {
        "created_unix": time.time(),
        "env": args.env,
        "out_dir": args.out_dir,
        "checkpoint_path": checkpoint_path,
        "normalization_path": os.path.join(args.out_dir, "normalization.npz"),
        "progress_path": progress_path,
        "hider_variables_path": hider_variables_path,
        "seeker_variables_path": seeker_variables_path,
        "obs_keys": obs_keys,
        "n_agents": n_agents,
        "n_hiders": n_hiders,
        "n_seekers": n_agents - n_hiders,
        "actor_dim": int(actor_example.shape[1]),
        "critic_dim": int(critic_example.shape[1]),
        "hidden_sizes": hidden_sizes,
        "updates_completed": args.updates,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "seed": args.seed,
        "last_progress": last_progress,
        "evaluation": evaluation,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as summary_file:
        json.dump(make_serializable(summary), summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    write_report(os.path.join(args.out_dir, "report.txt"), summary)

    print("saved_checkpoint:", checkpoint_path, flush=True)
    print("saved_progress:", progress_path, flush=True)
    print("saved_summary:", summary_path, flush=True)
    print("saved_report:", os.path.join(args.out_dir, "report.txt"), flush=True)


if __name__ == "__main__":
    main()
