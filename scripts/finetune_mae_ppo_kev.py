import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cyber_rl.benchmark import default_defenders, run_episode, summarize_results  # noqa: E402
from cyber_rl.env import (  # noqa: E402
    ATTACK_EXPLOIT_OFFSET,
    ATTACK_SCAN,
    ATTACK_WAIT,
    CyberHideSeekEnv,
)
from cyber_rl.kev import kev_families, load_kev_catalog, make_kev_scenario, normalize_catalog  # noqa: E402
from cyber_rl.policies import (  # noqa: E402
    AdaptiveDefender,
    DecoyFrontierDefender,
    GreedyAttacker,
    PatchHighValueDefender,
    RandomAttacker,
    RandomDefender,
    StealthAttacker,
    TargetedAttacker,
    valid_indices,
)
from evaluate_mae_ppo_kev_transfer import (  # noqa: E402
    MAEPPOProbe,
    MAETransferAttacker,
    MAETransferDefender,
    ROLE_HIDER,
    ROLE_SEEKER,
    cyber_concept_vector,
    load_linear_q,
    load_q_table,
)
from train_mae_ppo import make_serializable  # noqa: E402


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_json(path, payload):
    with open(path, "w") as output_file:
        json.dump(make_serializable(payload), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted(set(key for row in rows for key in row.keys()))
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(float(np.sum(exp)), 1e-8)


def discounted_returns(rewards, gamma):
    returns = np.zeros((len(rewards),), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns


def mae_stats_vector(stats):
    movement_expectation = [float(value) / 10.0 for value in stats["movement_expectation"]]
    movement_up = [float(value) for value in stats["movement_up"]]
    movement_down = [float(value) for value in stats["movement_down"]]
    pull = float(stats["pull"])
    glue = float(stats["glue"])
    return np.asarray(
        movement_expectation
        + movement_up
        + movement_down
        + [
            pull,
            glue,
            float(np.mean(movement_up)),
            float(np.mean(movement_down)),
            pull - glue,
            glue - pull,
        ],
        dtype=np.float32,
    )


def policy_features(obs, role_id, probe):
    base = cyber_concept_vector(obs, role_id, probe.actor_dim).reshape(-1)
    stats = mae_stats_vector(probe.policy_stats(obs, role_id))
    return np.concatenate([base.astype(np.float32), stats], axis=0)


def valid_mask(obs, role):
    if role == "attacker":
        return np.asarray(obs["valid_attacker_actions"], dtype=np.float32)
    return np.asarray(obs["valid_defender_actions"], dtype=np.float32)


def attacker_action_type_and_node(obs, action):
    n_nodes = len(obs["adjacency"])
    current = obs["current_node"]
    target = obs["target_node"]
    if action == 0:
        return 0, current
    if action == 1:
        return 1, current
    if action == 2:
        return 2, target
    if 3 <= action < 3 + n_nodes:
        return 3, action - 3
    return 4, action - 3 - n_nodes


class MAEFeatureQAttacker(object):
    name = "mae_ppo_kev_q_finetuned_attacker"

    def __init__(self, probe, teacher, n_nodes, max_steps=24, seed=0):
        self.probe = probe
        self.teacher = teacher
        self.n_nodes = int(n_nodes)
        self.max_steps = float(max_steps)
        self.rng = random.Random(seed)
        self.feature_dim = 45
        self.weights = np.zeros((self.feature_dim,), dtype=np.float32)
        self._initialize_from_transfer()

    def _initialize_from_transfer(self):
        # Bias the initial Q policy toward the MAE checkpoint-500 transfer action,
        # while still allowing KEV reward updates to override it.
        self.weights[0] = -0.2
        self.weights[1] = -2.0
        self.weights[2] = 0.5
        self.weights[6 + 2] = 2.0
        self.weights[6 + 3] = 1.0
        self.weights[20] = 2.5
        self.weights[21] = 1.5
        self.weights[22] = 1.0
        self.weights[23] = 2.0
        self.weights[37] = 3.0
        self.weights[38] = 1.0

    def reset(self):
        return None

    def _context(self, obs):
        stats = mae_stats_vector(self.probe.policy_stats(obs, ROLE_SEEKER))
        teacher_action = self.teacher.act(obs)
        teacher_type, teacher_node = attacker_action_type_and_node(obs, teacher_action)
        return stats, teacher_action, teacher_type, teacher_node

    def _features(self, obs, action, context=None):
        if context is None:
            context = self._context(obs)
        stats, teacher_action, teacher_type, teacher_node = context
        n = float(max(len(obs["adjacency"]), 1))
        current = obs["current_node"]
        target = obs["target_node"]
        current_dist = min(obs["dist_to_target"][current], len(obs["adjacency"])) / n
        compromised_frac = float(sum(1 for value in obs["compromised"] if value)) / n
        discovered_frac = float(sum(1 for value in obs["discovered"] if value)) / n
        detection = min(float(obs["detection"]) / 6.0, 1.5)
        steps_remaining = float(obs["steps_remaining"]) / max(self.max_steps, 1.0)
        action_type, node = attacker_action_type_and_node(obs, action)

        features = np.zeros((self.feature_dim,), dtype=np.float32)
        features[0] = 1.0
        features[1] = detection
        features[2] = steps_remaining
        features[3] = compromised_frac
        features[4] = discovered_frac
        features[5] = current_dist
        features[6 + action_type] = 1.0

        if 0 <= node < len(obs["adjacency"]):
            known_vuln = obs["known_vuln"][node]
            if known_vuln < 0:
                known_vuln = 0
            node_dist = min(obs["dist_to_target"][node], len(obs["adjacency"])) / n
            features[11] = node_dist
            features[12] = float(known_vuln) / 5.0
            features[13] = 1.0 if obs["discovered"][node] else 0.0
            features[14] = 1.0 if obs["compromised"][node] else 0.0
            features[15] = 1.0 if node == target else 0.0
            features[16] = min(float(obs["values"][node]) / 10.0, 1.5)
            features[17] = max(0.0, current_dist - node_dist)
            features[18] = 1.0 if obs["patched"][node] else 0.0
            features[19] = 1.0 if obs["decoys"][node] else 0.0

        features[20] = stats[9]  # MAE pull probability: intervention/exploit drive.
        features[21] = stats[10]  # MAE glue probability: caution/concealment drive.
        features[22] = stats[11]  # mean upward movement pressure.
        features[23] = stats[12]  # mean downward movement pressure.
        features[24] = stats[13]  # pull - glue.
        features[25] = stats[14]  # glue - pull.
        features[26:35] = stats[:9]
        features[35] = 1.0 if action == teacher_action else 0.0
        features[36] = 1.0 if action_type == teacher_type else 0.0
        features[37] = 1.0 if node == teacher_node else 0.0
        features[38] = 1.0 if obs["valid_attacker_actions"][2] else 0.0
        features[39] = detection if action == 0 else 0.0
        features[40] = 1.0 if action == 1 and not obs["scanned"][current] else 0.0
        features[41] = min(float(len(valid_indices(obs["valid_attacker_actions"]))) / float(max(len(obs["valid_attacker_actions"]), 1)), 1.0)
        features[42] = 1.0 if action_type in (3, 4) and node == target else 0.0
        features[43] = 1.0 if action_type == 4 and node in obs["adjacency"][current] else 0.0
        features[44] = 1.0 if action_type == 3 and node in obs["adjacency"][current] else 0.0
        return features

    def value(self, obs, action, context=None):
        return float(np.dot(self.weights, self._features(obs, action, context=context)))

    def act(self, obs, epsilon=0.0):
        valid = valid_indices(obs["valid_attacker_actions"])
        if not valid:
            return 0
        if self.rng.random() < epsilon:
            return self.rng.choice(valid)
        context = self._context(obs)
        best_action = valid[0]
        best_value = self.value(obs, best_action, context=context)
        for action in valid[1:]:
            value = self.value(obs, action, context=context)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def update(self, obs, action, reward, next_obs, done, alpha, gamma):
        context = self._context(obs)
        features = self._features(obs, action, context=context)
        target = float(reward) / 100.0
        if not done:
            next_valid = valid_indices(next_obs["valid_attacker_actions"])
            if next_valid:
                next_context = self._context(next_obs)
                target += gamma * max(self.value(next_obs, next_action, context=next_context) for next_action in next_valid)
        prediction = float(np.dot(self.weights, features))
        td_error = max(-5.0, min(5.0, target - prediction))
        self.weights += float(alpha) * td_error * features
        self.weights = np.clip(self.weights, -25.0, 25.0)
        return td_error

    def to_jsonable(self):
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "n_nodes": self.n_nodes,
            "max_steps": self.max_steps,
            "weights": self.weights.round(6).tolist(),
        }


class MaskedTFPolicy(object):
    def __init__(self, name, feature_dim, action_dim, hidden_sizes, learning_rate, entropy_coef, value_coef, seed):
        self.name = name
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = [int(value) for value in hidden_sizes]
        self.learning_rate = float(learning_rate)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            tf.set_random_seed(seed)
            self.x = tf.placeholder(tf.float32, shape=[None, self.feature_dim], name="x")
            self.valid = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="valid")
            self.actions = tf.placeholder(tf.int32, shape=[None], name="actions")
            self.teacher_actions = tf.placeholder(tf.int32, shape=[None], name="teacher_actions")
            self.advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")
            self.returns = tf.placeholder(tf.float32, shape=[None], name="returns")
            self.bc_weight = tf.placeholder(tf.float32, shape=[], name="bc_weight")

            hidden = self.x
            for idx, size in enumerate(self.hidden_sizes):
                hidden = tf.layers.dense(hidden, size, activation=tf.nn.tanh, name="dense_{}".format(idx))
            self.raw_logits = tf.layers.dense(hidden, self.action_dim, name="logits")
            self.value = tf.squeeze(tf.layers.dense(hidden, 1, name="value"), axis=1)
            self.masked_logits = self.raw_logits + (1.0 - self.valid) * -1e9
            self.probs = tf.nn.softmax(self.masked_logits)
            log_probs_all = tf.nn.log_softmax(self.masked_logits)
            one_hot = tf.one_hot(self.actions, self.action_dim)
            selected_log_probs = tf.reduce_sum(one_hot * log_probs_all, axis=1)

            self.entropy = -tf.reduce_mean(tf.reduce_sum(self.probs * log_probs_all, axis=1))
            self.bc_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.teacher_actions, logits=self.masked_logits)
            )
            self.policy_loss = -tf.reduce_mean(selected_log_probs * self.advantages)
            self.value_loss = tf.reduce_mean(tf.square(self.value - self.returns))
            self.rl_loss = (
                self.policy_loss
                + self.value_coef * self.value_loss
                - self.entropy_coef * self.entropy
                + self.bc_weight * self.bc_loss
            )

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.bc_train = optimizer.minimize(self.bc_loss)
            self.rl_train = optimizer.minimize(self.rl_loss)
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())

    def predict(self, features, masks):
        return self.session.run(
            [self.raw_logits, self.probs, self.value],
            feed_dict={self.x: features, self.valid: masks},
        )

    def act(self, obs, role, probe, rng, deterministic=False):
        features = policy_features(obs, ROLE_SEEKER if role == "attacker" else ROLE_HIDER, probe).reshape(1, -1)
        masks = valid_mask(obs, role).reshape(1, -1)
        logits, probs, _ = self.predict(features, masks)
        masked_logits = logits[0].astype(np.float64)
        for idx, flag in enumerate(masks[0]):
            if not flag:
                masked_logits[idx] = -1e9
        if deterministic:
            return int(np.argmax(masked_logits))
        distribution = softmax(masked_logits)
        return int(rng.choice(np.arange(self.action_dim), p=distribution))

    def values(self, features, masks):
        return self.predict(features, masks)[2]

    def train_bc(self, features, masks, actions, epochs, batch_size, rng):
        history = []
        count = len(actions)
        indices = np.arange(count)
        for epoch in range(epochs):
            rng.shuffle(indices)
            losses = []
            for start in range(0, count, batch_size):
                batch_idx = indices[start:start + batch_size]
                loss, _ = self.session.run(
                    [self.bc_loss, self.bc_train],
                    feed_dict={
                        self.x: features[batch_idx],
                        self.valid: masks[batch_idx],
                        self.actions: actions[batch_idx],
                        self.teacher_actions: actions[batch_idx],
                    },
                )
                losses.append(float(loss))
            history.append({"epoch": epoch + 1, "bc_loss": round(float(np.mean(losses)), 6)})
        return history

    def train_rl(self, features, masks, actions, teacher_actions, returns, bc_weight):
        values = self.values(features, masks).astype(np.float32)
        advantages = returns.astype(np.float32) - values
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / max(float(np.std(advantages)), 1e-6)
        loss, policy_loss, value_loss, entropy, bc_loss, _ = self.session.run(
            [self.rl_loss, self.policy_loss, self.value_loss, self.entropy, self.bc_loss, self.rl_train],
            feed_dict={
                self.x: features,
                self.valid: masks,
                self.actions: actions,
                self.teacher_actions: teacher_actions,
                self.advantages: advantages,
                self.returns: returns,
                self.bc_weight: float(bc_weight),
            },
        )
        return {
            "loss": round(float(loss), 6),
            "policy_loss": round(float(policy_loss), 6),
            "value_loss": round(float(value_loss), 6),
            "entropy": round(float(entropy), 6),
            "bc_loss": round(float(bc_loss), 6),
        }

    def save(self, path):
        ensure_dir(os.path.dirname(path))
        return self.saver.save(self.session, path)


class FineTunedPolicyWrapper(object):
    def __init__(self, model, role, probe, name, seed=0):
        self.model = model
        self.role = role
        self.probe = probe
        self.name = name
        self.rng = np.random.RandomState(seed)

    def reset(self):
        return None

    def act(self, obs):
        return self.model.act(obs, self.role, self.probe, self.rng, deterministic=True)


def collect_bc_dataset(catalog, families, role, probe, teacher, samples, seed, n_nodes, max_steps, exploration):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed + 77)
    features = []
    masks = []
    actions = []
    default_attackers = [RandomAttacker(seed=seed + 1), GreedyAttacker(), TargetedAttacker(), StealthAttacker()]
    default_defenders = [RandomDefender(seed=seed + 2), PatchHighValueDefender(), DecoyFrontierDefender(), AdaptiveDefender()]

    episode = 0
    while len(actions) < samples:
        family = families[episode % len(families)]
        scenario_seed = seed * 100000 + episode
        scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
        env = CyberHideSeekEnv(scenario, seed=scenario_seed + 19)
        obs = env.reset(seed=scenario_seed + 19)
        attacker = default_attackers[episode % len(default_attackers)]
        defender = default_defenders[episode % len(default_defenders)]
        attacker.reset()
        defender.reset()
        teacher.reset()
        done = False
        while not done and len(actions) < samples:
            if role == "attacker":
                label = teacher.act(obs)
                features.append(policy_features(obs, ROLE_SEEKER, probe))
                masks.append(valid_mask(obs, "attacker"))
                actions.append(label)
                if rng.random() < exploration:
                    attacker_action = rng.choice(valid_indices(obs["valid_attacker_actions"]))
                else:
                    attacker_action = label
                defender_action = defender.act(obs)
            else:
                label = teacher.act(obs)
                features.append(policy_features(obs, ROLE_HIDER, probe))
                masks.append(valid_mask(obs, "defender"))
                actions.append(label)
                attacker_action = attacker.act(obs)
                if rng.random() < exploration:
                    defender_action = rng.choice(valid_indices(obs["valid_defender_actions"]))
                else:
                    defender_action = label
            obs, _, done, _ = env.step(attacker_action, defender_action)
        episode += 1

    indices = np.arange(len(actions))
    np_rng.shuffle(indices)
    return (
        np.asarray(features, dtype=np.float32)[indices],
        np.asarray(masks, dtype=np.float32)[indices],
        np.asarray(actions, dtype=np.int32)[indices],
    )


def rollout_training_episode(
    catalog,
    families,
    episode,
    seed,
    n_nodes,
    max_steps,
    probe,
    attacker_model,
    defender_model,
    attacker_teacher,
    defender_teacher,
    mode,
):
    rng = np.random.RandomState(seed + episode * 13)
    family = families[episode % len(families)]
    scenario_seed = seed * 100000 + episode
    scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
    env = CyberHideSeekEnv(scenario, seed=scenario_seed + 31)
    obs = env.reset(seed=scenario_seed + 31)

    attacker_pool = [RandomAttacker(seed=seed + episode), GreedyAttacker(), TargetedAttacker(), StealthAttacker()]
    defender_pool = [RandomDefender(seed=seed + episode + 3), PatchHighValueDefender(), DecoyFrontierDefender(), AdaptiveDefender()]
    external_attacker = attacker_pool[episode % len(attacker_pool)]
    external_defender = defender_pool[episode % len(defender_pool)]
    external_attacker.reset()
    external_defender.reset()
    attacker_teacher.reset()
    defender_teacher.reset()

    attacker_batch = {"features": [], "masks": [], "actions": [], "rewards": [], "teacher_actions": []}
    defender_batch = {"features": [], "masks": [], "actions": [], "rewards": [], "teacher_actions": []}

    done = False
    while not done:
        if mode in ("attacker", "self_play"):
            attacker_features = policy_features(obs, ROLE_SEEKER, probe)
            attacker_mask = valid_mask(obs, "attacker")
            attacker_action = attacker_model.act(obs, "attacker", probe, rng, deterministic=False)
            attacker_batch["features"].append(attacker_features)
            attacker_batch["masks"].append(attacker_mask)
            attacker_batch["actions"].append(attacker_action)
            attacker_batch["teacher_actions"].append(attacker_teacher.act(obs))
        else:
            attacker_action = external_attacker.act(obs)

        if mode in ("defender", "self_play"):
            defender_features = policy_features(obs, ROLE_HIDER, probe)
            defender_mask = valid_mask(obs, "defender")
            defender_action = defender_model.act(obs, "defender", probe, rng, deterministic=False)
            defender_batch["features"].append(defender_features)
            defender_batch["masks"].append(defender_mask)
            defender_batch["actions"].append(defender_action)
            defender_batch["teacher_actions"].append(defender_teacher.act(obs))
        else:
            defender_action = external_defender.act(obs)

        obs, reward, done, info = env.step(attacker_action, defender_action)
        if mode in ("attacker", "self_play"):
            attacker_batch["rewards"].append(float(reward) / 100.0)
        if mode in ("defender", "self_play"):
            defender_batch["rewards"].append(float(-reward) / 100.0)

    return info, attacker_batch, defender_batch


def materialize_batch(batch, gamma):
    if not batch["actions"]:
        return None
    returns = discounted_returns(batch["rewards"], gamma)
    return {
        "features": np.asarray(batch["features"], dtype=np.float32),
        "masks": np.asarray(batch["masks"], dtype=np.float32),
        "actions": np.asarray(batch["actions"], dtype=np.int32),
        "teacher_actions": np.asarray(batch["teacher_actions"], dtype=np.int32),
        "returns": returns.astype(np.float32),
    }


def run_finetuning(
    catalog,
    families,
    probe,
    attacker_model,
    defender_model,
    attacker_teacher,
    defender_teacher,
    episodes,
    seed,
    n_nodes,
    max_steps,
    gamma,
    bc_weight_start,
    bc_weight_end,
):
    progress = []
    attacker_updates = []
    defender_updates = []
    for episode in range(episodes):
        if episode % 5 == 4:
            mode = "self_play"
        elif episode % 2 == 0:
            mode = "attacker"
        else:
            mode = "defender"
        info, attacker_batch, defender_batch = rollout_training_episode(
            catalog=catalog,
            families=families,
            episode=episode,
            seed=seed,
            n_nodes=n_nodes,
            max_steps=max_steps,
            probe=probe,
            attacker_model=attacker_model,
            defender_model=defender_model,
            attacker_teacher=attacker_teacher,
            defender_teacher=defender_teacher,
            mode=mode,
        )
        fraction = float(episode) / float(max(episodes - 1, 1))
        bc_weight = bc_weight_end + (bc_weight_start - bc_weight_end) * max(0.0, 1.0 - fraction)

        attacker_rl = None
        defender_rl = None
        attacker_ready = materialize_batch(attacker_batch, gamma)
        defender_ready = materialize_batch(defender_batch, gamma)
        if attacker_ready is not None:
            attacker_rl = attacker_model.train_rl(
                attacker_ready["features"],
                attacker_ready["masks"],
                attacker_ready["actions"],
                attacker_ready["teacher_actions"],
                attacker_ready["returns"],
                bc_weight=bc_weight,
            )
            attacker_updates.append(attacker_rl)
        if defender_ready is not None:
            defender_rl = defender_model.train_rl(
                defender_ready["features"],
                defender_ready["masks"],
                defender_ready["actions"],
                defender_ready["teacher_actions"],
                defender_ready["returns"],
                bc_weight=bc_weight,
            )
            defender_updates.append(defender_rl)

        row = {
            "episode": episode + 1,
            "mode": mode,
            "family": families[episode % len(families)],
            "outcome": info["outcome"],
            "attacker_success": int(info["attacker_success"]),
            "caught": int(info["caught"]),
            "timeout": int(info["timeout"]),
            "attacker_return": info["attacker_return"],
            "steps": info["steps"],
            "detection": info["detection"],
            "bc_weight": round(float(bc_weight), 6),
        }
        if attacker_rl:
            row.update({"attacker_" + key: value for key, value in attacker_rl.items()})
        if defender_rl:
            row.update({"defender_" + key: value for key, value in defender_rl.items()})
        progress.append(row)

        if (episode + 1) % max(1, episodes // 10) == 0:
            recent = progress[-max(1, min(50, len(progress))):]
            success = np.mean([item["attacker_success"] for item in recent])
            caught = np.mean([item["caught"] for item in recent])
            returns = np.mean([item["attacker_return"] for item in recent])
            print(
                "episode={} mode={} recent_success={:.3f} recent_caught={:.3f} recent_return={:.2f}".format(
                    episode + 1,
                    mode,
                    success,
                    caught,
                    returns,
                ),
                flush=True,
            )

    return {
        "progress": progress,
        "attacker_updates": attacker_updates,
        "defender_updates": defender_updates,
    }


def q_curriculum_defender(episode, episodes, seed):
    phase_size = max(1, int(episodes / 4))
    if episode < phase_size:
        pool = [PatchHighValueDefender()]
    elif episode < phase_size * 2:
        pool = [PatchHighValueDefender(), RandomDefender(seed=seed + 9)]
    elif episode < phase_size * 3:
        pool = [PatchHighValueDefender(), DecoyFrontierDefender(), RandomDefender(seed=seed + 13)]
    else:
        pool = [PatchHighValueDefender(), DecoyFrontierDefender(), AdaptiveDefender(), RandomDefender(seed=seed + 17)]
    return pool[(episode + seed) % len(pool)]


def train_mae_feature_q_attacker(
    catalog,
    families,
    probe,
    teacher,
    episodes,
    seed,
    n_nodes,
    max_steps,
    alpha,
    gamma,
    epsilon_start,
    epsilon_end,
):
    agent = MAEFeatureQAttacker(probe=probe, teacher=teacher, n_nodes=n_nodes, max_steps=max_steps, seed=seed + 1)
    history = []
    for episode in range(episodes):
        family = families[episode % len(families)]
        scenario_seed = seed * 100000 + episode
        scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
        env = CyberHideSeekEnv(scenario, seed=scenario_seed + 43)
        defender = q_curriculum_defender(episode, episodes, seed)
        defender.reset()
        obs = env.reset(seed=scenario_seed + 43)
        done = False
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - float(episode) / float(max(episodes - 1, 1)))
        td_errors = []
        while not done:
            defender_action = defender.act(obs)
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action, defender_action)
            td_errors.append(agent.update(obs, action, reward, next_obs, done, alpha=alpha, gamma=gamma))
            obs = next_obs
        if (episode + 1) % max(1, episodes // 20) == 0:
            history.append({
                "episode": episode + 1,
                "epsilon": round(float(epsilon), 6),
                "outcome": info["outcome"],
                "attacker_success": int(info["attacker_success"]),
                "caught": int(info["caught"]),
                "timeout": int(info["timeout"]),
                "attacker_return": info["attacker_return"],
                "weight_norm": round(float(np.linalg.norm(agent.weights)), 6),
                "mean_abs_td_error": round(float(np.mean(np.abs(td_errors))) if td_errors else 0.0, 6),
            })
    return agent, {
        "kind": "mae_feature_q_attacker",
        "episodes": episodes,
        "seed": seed,
        "families": list(families),
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "feature_dim": agent.feature_dim,
        "weight_norm": round(float(np.linalg.norm(agent.weights)), 6),
        "history": history,
    }


def best_compromised_distance(obs):
    distances = [
        obs["dist_to_target"][idx]
        for idx, compromised in enumerate(obs["compromised"])
        if compromised
    ]
    if not distances:
        return max(obs["dist_to_target"]) + 1
    return min(distances)


def shaped_success_reward(obs, action, reward, next_obs, info, targeted_action):
    current = obs["current_node"]
    next_current = next_obs["current_node"]
    current_progress = obs["dist_to_target"][current] - next_obs["dist_to_target"][next_current]
    compromised_progress = best_compromised_distance(obs) - best_compromised_distance(next_obs)
    discovered_delta = sum(1 for value in next_obs["discovered"] if value) - sum(1 for value in obs["discovered"] if value)
    detection_delta = float(next_obs["detection"]) - float(obs["detection"])

    shaped = float(reward) / 100.0
    shaped += 0.35 * max(0.0, float(current_progress))
    shaped += 0.30 * max(0.0, float(compromised_progress))
    shaped += 0.04 * max(0.0, float(discovered_delta))

    event_counts = info.get("event_counts", {})
    if event_counts.get("exploit_successes", 0) > 0 and compromised_progress > 0:
        shaped += 0.18
    if action == ATTACK_SCAN and discovered_delta <= 0:
        shaped -= 0.05
    if action == ATTACK_WAIT:
        if float(obs["detection"]) >= 4.8:
            shaped += 0.04
        else:
            shaped -= 0.10

    action_type, _ = attacker_action_type_and_node(obs, action)
    targeted_type, _ = attacker_action_type_and_node(obs, targeted_action)
    if action == targeted_action and float(obs["detection"]) < 4.5:
        shaped += 0.10
    elif action_type == targeted_type and float(obs["detection"]) < 4.0:
        shaped += 0.04

    if info["attacker_success"]:
        shaped += 1.50
    if info["timeout"]:
        shaped -= 0.85
    if info["caught"]:
        shaped -= 0.95

    if float(next_obs["detection"]) >= 4.5:
        shaped -= 0.12 * max(0.0, float(next_obs["detection"]) - 4.5)
    if detection_delta > 0.0 and float(next_obs["detection"]) >= 3.5:
        shaped -= 0.05 * detection_delta

    if action >= ATTACK_EXPLOIT_OFFSET and action < ATTACK_EXPLOIT_OFFSET + len(obs["adjacency"]):
        node = action - ATTACK_EXPLOIT_OFFSET
        if obs["patched"][node]:
            shaped -= 0.05
        if obs["decoys"][node]:
            shaped -= 0.30

    return shaped


def train_success_first_mae_targeted_q_attacker(
    catalog,
    families,
    probe,
    episodes,
    seed,
    n_nodes,
    max_steps,
    alpha,
    gamma,
    epsilon_start,
    epsilon_end,
):
    targeted_teacher = TargetedAttacker()
    agent = MAEFeatureQAttacker(probe=probe, teacher=targeted_teacher, n_nodes=n_nodes, max_steps=max_steps, seed=seed + 1)
    agent.name = "mae_ppo_targeted_q_finetuned_attacker"
    agent.weights[1] = -0.35
    agent.weights[6 + 1] = 0.35
    agent.weights[6 + 2] = 4.00
    agent.weights[6 + 3] = 1.75
    agent.weights[6 + 4] = 1.75
    agent.weights[17] = 2.50
    agent.weights[20] = 1.25
    agent.weights[21] = 0.35
    agent.weights[35] = 3.00
    agent.weights[36] = 1.20
    agent.weights[37] = 2.50
    agent.weights[42] = 2.00
    agent.weights[43] = 0.80
    agent.weights[44] = 0.80

    history = []
    for episode in range(episodes):
        family = families[episode % len(families)]
        scenario_seed = seed * 100000 + episode
        scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
        env = CyberHideSeekEnv(scenario, seed=scenario_seed + 59)
        defender = q_curriculum_defender(episode, episodes, seed + 37)
        defender.reset()
        targeted_teacher.reset()
        obs = env.reset(seed=scenario_seed + 59)
        done = False
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - float(episode) / float(max(episodes - 1, 1)))
        td_errors = []
        shaped_rewards = []
        while not done:
            defender_action = defender.act(obs)
            targeted_action = targeted_teacher.act(obs)
            if random.Random(seed + episode * 997 + len(td_errors)).random() < 0.12 and obs["detection"] < 4.0:
                action = targeted_action
            else:
                action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action, defender_action)
            shaped_reward = shaped_success_reward(obs, action, reward, next_obs, info, targeted_action)
            shaped_rewards.append(shaped_reward)
            td_errors.append(agent.update(obs, action, shaped_reward * 100.0, next_obs, done, alpha=alpha, gamma=gamma))
            obs = next_obs
        if (episode + 1) % max(1, episodes // 20) == 0:
            history.append({
                "episode": episode + 1,
                "epsilon": round(float(epsilon), 6),
                "outcome": info["outcome"],
                "attacker_success": int(info["attacker_success"]),
                "caught": int(info["caught"]),
                "timeout": int(info["timeout"]),
                "attacker_return": info["attacker_return"],
                "weight_norm": round(float(np.linalg.norm(agent.weights)), 6),
                "mean_abs_td_error": round(float(np.mean(np.abs(td_errors))) if td_errors else 0.0, 6),
                "mean_shaped_reward": round(float(np.mean(shaped_rewards)) if shaped_rewards else 0.0, 6),
            })
    return agent, {
        "kind": "success_first_mae_targeted_q_attacker",
        "episodes": episodes,
        "seed": seed,
        "families": list(families),
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "feature_dim": agent.feature_dim,
        "weight_norm": round(float(np.linalg.norm(agent.weights)), 6),
        "history": history,
    }


def run_suite(catalog, attackers, defenders, families, episodes_per_family, seed, n_nodes, max_steps):
    rows = []
    pair_summaries = []
    for attacker in attackers:
        for defender in defenders:
            pair_records = []
            for family_idx, family in enumerate(families):
                for episode_idx in range(episodes_per_family):
                    scenario_seed = seed + family_idx * 10000 + episode_idx
                    eval_seed = seed * 1000000 + family_idx * 10000 + episode_idx
                    scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
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
        "created_unix": time.time(),
        "dataset": {
            "title": normalized["title"],
            "catalogVersion": normalized["catalogVersion"],
            "dateReleased": normalized["dateReleased"],
            "count": normalized["count"],
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
    }


def write_report(path, payload):
    suite = payload["evaluation"]
    lines = [
        "# MAE PPO Checkpoint-500 KEV Fine-Tuning",
        "",
        "- source_checkpoint: {}".format(payload["source"]["checkpoint"]),
        "- source_normalization: {}".format(payload["source"]["normalization"]),
        "- transfer_type: MAE PPO checkpoint stats plus KEV policy-gradient fine-tuning",
        "- finetune_episodes: {}".format(payload["training"]["episodes"]),
        "- mae_feature_q_episodes: {}".format(payload["training"]["mae_q_summary"]["episodes"]),
        "- success_first_q_episodes: {}".format(payload["training"]["success_q_summary"]["episodes"]),
        "- bc_samples_per_role: {}".format(payload["training"]["bc_samples_per_role"]),
        "- evaluation_episodes_per_family: {}".format(suite["episodes_per_family"]),
        "- families: {}".format(", ".join(suite["families"])),
        "",
        "## Training Summary",
        "",
    ]
    progress = payload["training"]["progress"]
    if progress:
        recent = progress[-max(1, min(50, len(progress))):]
        lines.extend([
            "Recent 50 training episodes:",
            "",
            "```text",
            "attacker_success: {:.3f}".format(float(np.mean([row["attacker_success"] for row in recent]))),
            "caught: {:.3f}".format(float(np.mean([row["caught"] for row in recent]))),
            "timeout: {:.3f}".format(float(np.mean([row["timeout"] for row in recent]))),
            "mean_attacker_return: {:.2f}".format(float(np.mean([row["attacker_return"] for row in recent]))),
            "```",
            "",
        ])
    lines.extend([
        "## Aggregate By Attacker",
        "",
        "| attacker | success | caught | timeout | return | steps |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(suite["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        lines.append("| {attacker} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} | {mean_steps:.2f} |".format(**row))
    lines.extend([
        "",
        "## Aggregate By Defender",
        "",
        "| defender | attacker success | caught | timeout | return | steps |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(suite["aggregate_by_defender"], key=lambda item: item["attacker_success_rate"]):
        lines.append("| {defender} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} | {mean_steps:.2f} |".format(**row))
    lines.extend([
        "",
        "## Pair Summary",
        "",
        "| attacker | defender | success | caught | timeout | return |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in sorted(suite["pair_summaries"], key=lambda item: (item["attacker"], item["defender"])):
        lines.append("| {attacker} | {defender} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} |".format(**row))
    with open(path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def write_progress_svg(path, progress):
    if not progress:
        return
    width = 960
    height = 420
    margin_left = 70
    margin_right = 30
    margin_top = 35
    margin_bottom = 55
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    window = 25

    def rolling(key):
        values = []
        for idx in range(len(progress)):
            start = max(0, idx + 1 - window)
            values.append(float(np.mean([row[key] for row in progress[start:idx + 1]])))
        return values

    series = [
        ("success", rolling("attacker_success"), "#1f77b4"),
        ("caught", rolling("caught"), "#d62728"),
        ("timeout", rolling("timeout"), "#7f7f7f"),
    ]

    def point(idx, value):
        x = margin_left + plot_w * float(idx) / float(max(len(progress) - 1, 1))
        y = margin_top + plot_h * (1.0 - max(0.0, min(1.0, value)))
        return x, y

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="24" font-family="Arial" font-size="18" font-weight="bold">KEV fine-tuning rolling outcome rates</text>'.format(margin_left),
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#222"/>'.format(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h),
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#222"/>'.format(margin_left, margin_top, margin_left, margin_top + plot_h),
    ]
    for tick in range(0, 6):
        value = tick / 5.0
        y = margin_top + plot_h * (1.0 - value)
        lines.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#ddd"/>'.format(margin_left, y, margin_left + plot_w, y))
        lines.append('<text x="20" y="{:.1f}" font-family="Arial" font-size="12">{:.1f}</text>'.format(y + 4, value))
    for name, values, color in series:
        points = []
        for idx, value in enumerate(values):
            x, y = point(idx, value)
            points.append("{:.1f},{:.1f}".format(x, y))
        lines.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(color, " ".join(points)))
    legend_x = margin_left + plot_w - 210
    for idx, (name, _, color) in enumerate(series):
        y = margin_top + 20 + idx * 22
        lines.append('<rect x="{}" y="{}" width="14" height="14" fill="{}"/>'.format(legend_x, y - 11, color))
        lines.append('<text x="{}" y="{}" font-family="Arial" font-size="13">{}</text>'.format(legend_x + 20, y, name))
    lines.append('<text x="{}" y="{}" font-family="Arial" font-size="12">episode</text>'.format(margin_left + plot_w / 2 - 20, height - 15))
    lines.append('</svg>')
    with open(path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kev-json", default="/workspace/data/raw/known_exploited_vulnerabilities.json")
    parser.add_argument("--checkpoint", default="/workspace/runs/mae_ppo_full_v1/model.ckpt-500")
    parser.add_argument("--normalization", default="/workspace/runs/mae_ppo_full_v1/normalization.npz")
    parser.add_argument("--train-summary", default="/workspace/runs/mae_ppo_full_v1/summary.json")
    parser.add_argument("--q-table-json", default="/workspace/runs/kev_realworld_benchmark_v1/q_table_kev.json")
    parser.add_argument("--linear-q-json", default="/workspace/runs/kev_realworld_benchmark_v1/linear_q_kev_weights.json")
    parser.add_argument("--out-dir", default="/workspace/runs/mae_ppo_full500_kev_finetune_v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--bc-samples", type=int, default=4000)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-sizes", default="192,192")
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--bc-weight-start", type=float, default=0.15)
    parser.add_argument("--bc-weight-end", type=float, default=0.02)
    parser.add_argument("--bc-exploration", type=float, default=0.15)
    parser.add_argument("--mae-q-episodes", type=int, default=-1)
    parser.add_argument("--mae-q-alpha", type=float, default=0.04)
    parser.add_argument("--mae-q-epsilon-start", type=float, default=0.30)
    parser.add_argument("--mae-q-epsilon-end", type=float, default=0.04)
    parser.add_argument("--success-q-episodes", type=int, default=0)
    parser.add_argument("--success-q-alpha", type=float, default=0.025)
    parser.add_argument("--success-q-epsilon-start", type=float, default=0.24)
    parser.add_argument("--success-q-epsilon-end", type=float, default=0.03)
    parser.add_argument("--episodes-per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--families", default=",".join(kev_families()))
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    catalog = load_kev_catalog(args.kev_json)
    families = [part.strip() for part in args.families.split(",") if part.strip()]
    hidden_sizes = [int(part.strip()) for part in args.hidden_sizes.split(",") if part.strip()]
    np_rng = np.random.RandomState(args.seed)

    probe = MAEPPOProbe(args.checkpoint, args.normalization, args.train_summary)
    attacker_teacher = MAETransferAttacker(probe)
    defender_teacher = MAETransferDefender(probe)

    probe_scenario = make_kev_scenario(catalog, families[0], args.seed, n_nodes=args.n_nodes, max_steps=args.max_steps)
    probe_env = CyberHideSeekEnv(probe_scenario, seed=args.seed)
    probe_obs = probe_env.reset(seed=args.seed)
    feature_dim = int(policy_features(probe_obs, ROLE_SEEKER, probe).shape[0])
    attacker_action_dim = int(probe_env.attack_action_count)
    defender_action_dim = int(probe_env.defend_action_count)

    attacker_model = MaskedTFPolicy(
        name="mae_ppo_kev_finetuned_attacker",
        feature_dim=feature_dim,
        action_dim=attacker_action_dim,
        hidden_sizes=hidden_sizes,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        seed=args.seed + 1,
    )
    defender_model = MaskedTFPolicy(
        name="mae_ppo_kev_finetuned_defender",
        feature_dim=feature_dim,
        action_dim=defender_action_dim,
        hidden_sizes=hidden_sizes,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        seed=args.seed + 2,
    )

    print("collecting_bc_attacker samples={}".format(args.bc_samples), flush=True)
    attacker_x, attacker_mask, attacker_y = collect_bc_dataset(
        catalog=catalog,
        families=families,
        role="attacker",
        probe=probe,
        teacher=attacker_teacher,
        samples=args.bc_samples,
        seed=args.seed + 10,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
        exploration=args.bc_exploration,
    )
    print("collecting_bc_defender samples={}".format(args.bc_samples), flush=True)
    defender_x, defender_mask, defender_y = collect_bc_dataset(
        catalog=catalog,
        families=families,
        role="defender",
        probe=probe,
        teacher=defender_teacher,
        samples=args.bc_samples,
        seed=args.seed + 20,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
        exploration=args.bc_exploration,
    )

    print("training_bc_attacker", flush=True)
    attacker_bc_history = attacker_model.train_bc(attacker_x, attacker_mask, attacker_y, args.bc_epochs, args.batch_size, np_rng)
    print("training_bc_defender", flush=True)
    defender_bc_history = defender_model.train_bc(defender_x, defender_mask, defender_y, args.bc_epochs, args.batch_size, np_rng)

    print("finetuning_kev episodes={}".format(args.episodes), flush=True)
    finetune = run_finetuning(
        catalog=catalog,
        families=families,
        probe=probe,
        attacker_model=attacker_model,
        defender_model=defender_model,
        attacker_teacher=attacker_teacher,
        defender_teacher=defender_teacher,
        episodes=args.episodes,
        seed=args.seed + 100,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        bc_weight_start=args.bc_weight_start,
        bc_weight_end=args.bc_weight_end,
    )

    mae_q_episodes = args.episodes if args.mae_q_episodes < 0 else args.mae_q_episodes
    print("training_mae_feature_q_attacker episodes={}".format(mae_q_episodes), flush=True)
    mae_q_attacker, mae_q_summary = train_mae_feature_q_attacker(
        catalog=catalog,
        families=families,
        probe=probe,
        teacher=attacker_teacher,
        episodes=mae_q_episodes,
        seed=args.seed + 500,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
        alpha=args.mae_q_alpha,
        gamma=args.gamma,
        epsilon_start=args.mae_q_epsilon_start,
        epsilon_end=args.mae_q_epsilon_end,
    )
    success_q_episodes = args.episodes if args.success_q_episodes < 0 else args.success_q_episodes
    print("training_success_first_mae_targeted_q_attacker episodes={}".format(success_q_episodes), flush=True)
    success_q_attacker, success_q_summary = train_success_first_mae_targeted_q_attacker(
        catalog=catalog,
        families=families,
        probe=probe,
        episodes=success_q_episodes,
        seed=args.seed + 700,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
        alpha=args.success_q_alpha,
        gamma=args.gamma,
        epsilon_start=args.success_q_epsilon_start,
        epsilon_end=args.success_q_epsilon_end,
    )

    attacker_ckpt = attacker_model.save(os.path.join(args.out_dir, "attacker_policy.ckpt"))
    defender_ckpt = defender_model.save(os.path.join(args.out_dir, "defender_policy.ckpt"))

    fine_attacker = FineTunedPolicyWrapper(
        attacker_model,
        "attacker",
        probe,
        "mae_ppo_kev_finetuned_attacker",
        seed=args.seed + 300,
    )
    fine_defender = FineTunedPolicyWrapper(
        defender_model,
        "defender",
        probe,
        "mae_ppo_kev_finetuned_defender",
        seed=args.seed + 301,
    )
    zero_shot_attacker = MAETransferAttacker(probe)
    zero_shot_defender = MAETransferDefender(probe)

    attackers = [
        RandomAttacker(seed=101),
        GreedyAttacker(),
        TargetedAttacker(),
        StealthAttacker(),
        load_q_table(args.q_table_json, seed=args.seed + 1),
        load_linear_q(args.linear_q_json, seed=args.seed + 2),
        zero_shot_attacker,
        mae_q_attacker,
        success_q_attacker,
        fine_attacker,
    ]
    defenders = default_defenders() + [zero_shot_defender, fine_defender]

    print("evaluating_finetuned", flush=True)
    evaluation = run_suite(
        catalog=catalog,
        attackers=attackers,
        defenders=defenders,
        families=families,
        episodes_per_family=args.episodes_per_family,
        seed=args.seed + 200,
        n_nodes=args.n_nodes,
        max_steps=args.max_steps,
    )

    payload = {
        "created_unix": time.time(),
        "source": {
            "checkpoint": args.checkpoint,
            "normalization": args.normalization,
            "train_summary": args.train_summary,
            "q_table_json": args.q_table_json,
            "linear_q_json": args.linear_q_json,
        },
        "model": {
            "feature_dim": feature_dim,
            "attacker_action_dim": attacker_action_dim,
            "defender_action_dim": defender_action_dim,
            "hidden_sizes": hidden_sizes,
            "attacker_checkpoint": attacker_ckpt,
            "defender_checkpoint": defender_ckpt,
        },
        "training": {
            "episodes": args.episodes,
            "bc_samples_per_role": args.bc_samples,
            "bc_epochs": args.bc_epochs,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "bc_weight_start": args.bc_weight_start,
            "bc_weight_end": args.bc_weight_end,
            "mae_q_summary": mae_q_summary,
            "success_q_summary": success_q_summary,
            "attacker_bc_history": attacker_bc_history,
            "defender_bc_history": defender_bc_history,
            "progress": finetune["progress"],
        },
        "evaluation": evaluation,
    }

    write_json(os.path.join(args.out_dir, "summary.json"), payload)
    write_json(os.path.join(args.out_dir, "mae_feature_q_attacker.json"), mae_q_attacker.to_jsonable())
    write_json(os.path.join(args.out_dir, "success_first_mae_targeted_q_attacker.json"), success_q_attacker.to_jsonable())
    write_csv(os.path.join(args.out_dir, "progress.csv"), finetune["progress"])
    write_csv(os.path.join(args.out_dir, "evaluation_episodes.csv"), evaluation["rows"])
    write_csv(os.path.join(args.out_dir, "evaluation_pair_summaries.csv"), evaluation["pair_summaries"])
    write_report(os.path.join(args.out_dir, "report.md"), payload)
    write_progress_svg(os.path.join(args.out_dir, "finetune_progress.svg"), finetune["progress"])

    print("saved_dir:", args.out_dir, flush=True)
    for row in sorted(evaluation["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        print(
            "attacker={attacker} success={attacker_success_rate:.3f} caught={caught_rate:.3f} timeout={timeout_rate:.3f} return={mean_attacker_return:.2f}".format(**row),
            flush=True,
        )
    for row in sorted(evaluation["aggregate_by_defender"], key=lambda item: item["attacker_success_rate"]):
        print(
            "defender={defender} attacker_success={attacker_success_rate:.3f} caught={caught_rate:.3f} timeout={timeout_rate:.3f}".format(**row),
            flush=True,
        )


if __name__ == "__main__":
    main()
