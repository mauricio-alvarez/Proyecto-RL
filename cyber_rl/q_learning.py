import json
import random

import numpy as np

from cyber_rl.env import CyberHideSeekEnv, make_scenario
from cyber_rl.policies import (
    AdaptiveDefender,
    DecoyFrontierDefender,
    NoopDefender,
    PatchHighValueDefender,
    RandomDefender,
    valid_indices,
)


def _bucket_detection(value):
    if value < 1.5:
        return 0
    if value < 3.0:
        return 1
    if value < 4.5:
        return 2
    return 3


def _mask(flags):
    value = 0
    for idx, flag in enumerate(flags):
        if flag:
            value |= 1 << idx
    return value


def state_key(obs):
    known = []
    for value in obs["known_vuln"]:
        if value < 0:
            known.append(0)
        elif value <= 2:
            known.append(1)
        elif value <= 4:
            known.append(2)
        else:
            known.append(3)
    return (
        obs["current_node"],
        obs["target_node"],
        _mask(obs["discovered"]),
        _mask(obs["compromised"]),
        _mask(obs["scanned"]),
        _bucket_detection(obs["detection"]),
        tuple(known),
    )


class QTableAttacker(object):
    name = "q_table"

    def __init__(self, action_count, seed=0):
        self.action_count = int(action_count)
        self.rng = random.Random(seed)
        self.q = {}

    def reset(self):
        return None

    def _values(self, key):
        if key not in self.q:
            self.q[key] = np.zeros(self.action_count, dtype=np.float32)
        return self.q[key]

    def act(self, obs, epsilon=0.0):
        key = state_key(obs)
        valid = valid_indices(obs["valid_attacker_actions"])
        if not valid:
            return 0
        if self.rng.random() < epsilon:
            return self.rng.choice(valid)
        values = self._values(key)
        best_action = valid[0]
        best_value = values[best_action]
        for action in valid[1:]:
            if values[action] > best_value:
                best_action = action
                best_value = values[action]
        return best_action

    def update(self, obs, action, reward, next_obs, done, alpha, gamma):
        key = state_key(obs)
        values = self._values(key)
        target = float(reward)
        if not done:
            next_key = state_key(next_obs)
            next_values = self._values(next_key)
            valid_next = valid_indices(next_obs["valid_attacker_actions"])
            if valid_next:
                target += gamma * float(np.max(next_values[valid_next]))
        values[action] += alpha * (target - values[action])

    def to_jsonable(self):
        items = []
        for key, values in self.q.items():
            items.append({"key": json.dumps(key), "values": values.round(6).tolist()})
        return {"name": self.name, "action_count": self.action_count, "q": items}

    @classmethod
    def from_jsonable(cls, payload, seed=0):
        agent = cls(payload["action_count"], seed=seed)
        for item in payload["q"]:
            key = json.loads(item["key"])
            key = (
                key[0],
                key[1],
                key[2],
                key[3],
                key[4],
                key[5],
                tuple(key[6]),
            )
            agent.q[key] = np.asarray(item["values"], dtype=np.float32)
        return agent


class LinearQAttacker(object):
    name = "linear_q"

    def __init__(self, action_count, n_nodes, max_steps=24, seed=0):
        self.action_count = int(action_count)
        self.n_nodes = int(n_nodes)
        self.max_steps = float(max_steps)
        self.rng = random.Random(seed)
        self.weights = np.zeros(20, dtype=np.float32)

    def reset(self):
        return None

    def _features(self, obs, action):
        n = float(max(len(obs["adjacency"]), 1))
        current = obs["current_node"]
        target = obs["target_node"]
        current_dist = min(obs["dist_to_target"][current], len(obs["adjacency"])) / n
        compromised_frac = float(sum(1 for value in obs["compromised"] if value)) / n
        discovered_frac = float(sum(1 for value in obs["discovered"] if value)) / n
        detection = min(float(obs["detection"]) / 6.0, 1.5)
        steps_remaining = float(obs["steps_remaining"]) / max(self.max_steps, 1.0)

        features = np.zeros(20, dtype=np.float32)
        features[0] = 1.0
        features[1] = detection
        features[2] = steps_remaining
        features[3] = compromised_frac
        features[4] = discovered_frac
        features[5] = current_dist

        action_type = 0
        node = current
        if action == 0:
            action_type = 0
        elif action == 1:
            action_type = 1
        elif action == 2:
            action_type = 2
            node = target
        elif 3 <= action < 3 + len(obs["adjacency"]):
            action_type = 3
            node = action - 3
        else:
            action_type = 4
            node = action - 3 - len(obs["adjacency"])
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

        features[18] = detection if action == 0 else 0.0
        features[19] = 1.0 if action == 1 and not obs["scanned"][current] else 0.0
        return features

    def value(self, obs, action):
        return float(np.dot(self.weights, self._features(obs, action)))

    def act(self, obs, epsilon=0.0):
        valid = valid_indices(obs["valid_attacker_actions"])
        if not valid:
            return 0
        if self.rng.random() < epsilon:
            return self.rng.choice(valid)
        best_action = valid[0]
        best_value = self.value(obs, best_action)
        for action in valid[1:]:
            value = self.value(obs, action)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def update(self, obs, action, reward, next_obs, done, alpha, gamma):
        features = self._features(obs, action)
        target = float(reward)
        if not done:
            valid_next = valid_indices(next_obs["valid_attacker_actions"])
            if valid_next:
                target += gamma * max(self.value(next_obs, next_action) for next_action in valid_next)
        prediction = float(np.dot(self.weights, features))
        td_error = max(-200.0, min(200.0, target - prediction))
        self.weights += alpha * td_error * features
        self.weights = np.clip(self.weights, -100.0, 100.0)

    def to_jsonable(self):
        return {
            "name": self.name,
            "action_count": self.action_count,
            "n_nodes": self.n_nodes,
            "max_steps": self.max_steps,
            "weights": self.weights.round(6).tolist(),
        }


def _curriculum_defender(episode, episodes, seed):
    curriculum = [
        [NoopDefender()],
        [NoopDefender(), PatchHighValueDefender()],
        [PatchHighValueDefender(), RandomDefender(seed=seed + 9)],
        [PatchHighValueDefender(), DecoyFrontierDefender(), AdaptiveDefender()],
    ]
    phase = min(len(curriculum) - 1, int(float(episode) / float(max(episodes, 1)) * len(curriculum)))
    defenders = curriculum[phase]
    return defenders[episode % len(defenders)], phase


def train_q_attacker(
    episodes=4000,
    seed=123,
    families=None,
    n_nodes=8,
    max_steps=24,
    alpha=0.15,
    gamma=0.97,
    epsilon_start=0.35,
    epsilon_end=0.04,
):
    if families is None:
        families = ["chain", "branching", "dense", "decoy_heavy", "random"]

    rng = random.Random(seed)
    probe = CyberHideSeekEnv(make_scenario(families[0], seed, n_nodes=n_nodes, max_steps=max_steps), seed=seed)
    agent = QTableAttacker(probe.attack_action_count, seed=seed + 1)
    history = []

    for episode in range(int(episodes)):
        family = families[episode % len(families)]
        scenario_seed = seed * 100000 + episode
        scenario = make_scenario(family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
        env = CyberHideSeekEnv(scenario, seed=scenario_seed)
        defender, phase = _curriculum_defender(episode, episodes, seed)
        obs = env.reset(seed=scenario_seed)
        agent.reset()
        defender.reset()
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - float(episode) / float(max(episodes - 1, 1)))

        done = False
        while not done:
            defender_action = defender.act(obs)
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action, defender_action)
            agent.update(obs, action, reward, next_obs, done, alpha=alpha, gamma=gamma)
            obs = next_obs

        if (episode + 1) % max(1, episodes // 20) == 0:
            history.append({
                "episode": episode + 1,
                "epsilon": round(epsilon, 4),
                "phase": phase,
                "q_states": len(agent.q),
                "last_outcome": info["outcome"],
                "last_return": info["attacker_return"],
            })

    return agent, {
        "episodes": int(episodes),
        "seed": seed,
        "families": families,
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "q_states": len(agent.q),
        "history": history,
    }


def train_linear_q_attacker(
    episodes=8000,
    seed=321,
    families=None,
    n_nodes=8,
    max_steps=24,
    alpha=0.02,
    gamma=0.97,
    epsilon_start=0.45,
    epsilon_end=0.05,
):
    if families is None:
        families = ["chain", "branching", "dense", "decoy_heavy", "random"]

    probe = CyberHideSeekEnv(make_scenario(families[0], seed, n_nodes=n_nodes, max_steps=max_steps), seed=seed)
    agent = LinearQAttacker(probe.attack_action_count, n_nodes=n_nodes, max_steps=max_steps, seed=seed + 1)
    history = []

    for episode in range(int(episodes)):
        family = families[episode % len(families)]
        scenario_seed = seed * 100000 + episode
        scenario = make_scenario(family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
        env = CyberHideSeekEnv(scenario, seed=scenario_seed)
        defender, phase = _curriculum_defender(episode, episodes, seed)
        obs = env.reset(seed=scenario_seed)
        agent.reset()
        defender.reset()
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - float(episode) / float(max(episodes - 1, 1)))

        done = False
        while not done:
            defender_action = defender.act(obs)
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action, defender_action)
            agent.update(obs, action, reward, next_obs, done, alpha=alpha, gamma=gamma)
            obs = next_obs

        if (episode + 1) % max(1, episodes // 20) == 0:
            history.append({
                "episode": episode + 1,
                "epsilon": round(epsilon, 4),
                "phase": phase,
                "last_outcome": info["outcome"],
                "last_return": info["attacker_return"],
                "weight_norm": round(float(np.linalg.norm(agent.weights)), 4),
            })

    return agent, {
        "episodes": int(episodes),
        "seed": seed,
        "families": families,
        "n_nodes": n_nodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "history": history,
        "weight_norm": round(float(np.linalg.norm(agent.weights)), 4),
    }
