import random
from collections import deque

from cyber_rl.env import (
    ATTACK_EXFILTRATE,
    ATTACK_EXPLOIT_OFFSET,
    ATTACK_SCAN,
    ATTACK_WAIT,
    DEFEND_NOOP,
)


def valid_indices(mask):
    return [idx for idx, flag in enumerate(mask) if flag]


def shortest_path_next(adjacency, start, target, allowed):
    if start == target:
        return start
    queue = deque([start])
    previous = {start: None}
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in allowed or nxt in previous:
                continue
            previous[nxt] = node
            if nxt == target:
                cur = target
                while previous[cur] != start:
                    cur = previous[cur]
                return cur
            queue.append(nxt)
    return None


class BasePolicy(object):
    name = "base"

    def reset(self):
        return None


class RandomAttacker(BasePolicy):
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs):
        choices = valid_indices(obs["valid_attacker_actions"])
        return self.rng.choice(choices)


class GreedyAttacker(BasePolicy):
    name = "greedy"

    def act(self, obs):
        n_nodes = len(obs["adjacency"])
        current = obs["current_node"]
        target = obs["target_node"]
        valid = obs["valid_attacker_actions"]
        if valid[ATTACK_EXFILTRATE]:
            return ATTACK_EXFILTRATE
        if not obs["scanned"][current]:
            return ATTACK_SCAN

        exploit_candidates = []
        for node in obs["adjacency"][current]:
            action = ATTACK_EXPLOIT_OFFSET + node
            if valid[action]:
                known_vuln = obs["known_vuln"][node]
                dist = obs["dist_to_target"][node]
                exploit_candidates.append((known_vuln, -dist, obs["values"][node], action))
        if exploit_candidates:
            exploit_candidates.sort(reverse=True)
            return exploit_candidates[0][-1]

        compromised_nodes = set(idx for idx, value in enumerate(obs["compromised"]) if value)
        next_node = shortest_path_next(obs["adjacency"], current, target, compromised_nodes)
        if next_node is not None:
            action = ATTACK_EXPLOIT_OFFSET + n_nodes + next_node
            if valid[action]:
                return action

        if valid[ATTACK_SCAN]:
            return ATTACK_SCAN
        return ATTACK_WAIT


class TargetedAttacker(BasePolicy):
    name = "targeted"

    def act(self, obs):
        n_nodes = len(obs["adjacency"])
        current = obs["current_node"]
        target = obs["target_node"]
        valid = obs["valid_attacker_actions"]
        if valid[ATTACK_EXFILTRATE]:
            return ATTACK_EXFILTRATE
        if not obs["scanned"][current]:
            return ATTACK_SCAN

        move_candidates = []
        for node in obs["adjacency"][current]:
            action = ATTACK_EXPLOIT_OFFSET + n_nodes + node
            if valid[action]:
                move_candidates.append((obs["dist_to_target"][node], action))
        if move_candidates:
            move_candidates.sort()
            if move_candidates[0][0] <= obs["dist_to_target"][current]:
                return move_candidates[0][1]

        exploit_candidates = []
        for node in obs["adjacency"][current]:
            action = ATTACK_EXPLOIT_OFFSET + node
            if valid[action]:
                exploit_candidates.append((obs["dist_to_target"][node], -obs["known_vuln"][node], action))
        if exploit_candidates:
            exploit_candidates.sort()
            return exploit_candidates[0][-1]

        if move_candidates:
            return move_candidates[0][1]
        return ATTACK_SCAN if valid[ATTACK_SCAN] else ATTACK_WAIT


class StealthAttacker(BasePolicy):
    name = "stealth"

    def act(self, obs):
        n_nodes = len(obs["adjacency"])
        current = obs["current_node"]
        target = obs["target_node"]
        valid = obs["valid_attacker_actions"]
        if valid[ATTACK_EXFILTRATE]:
            return ATTACK_EXFILTRATE
        if obs["detection"] >= 4.5:
            return ATTACK_WAIT
        if not obs["scanned"][current]:
            return ATTACK_SCAN

        exploit_candidates = []
        for node in obs["adjacency"][current]:
            action = ATTACK_EXPLOIT_OFFSET + node
            if valid[action]:
                known_vuln = obs["known_vuln"][node]
                dist = obs["dist_to_target"][node]
                value = obs["values"][node]
                risk = 0.5 if known_vuln >= 4 else 0.0
                exploit_candidates.append((-risk, -dist, value, known_vuln, action))
        if exploit_candidates:
            exploit_candidates.sort(reverse=True)
            return exploit_candidates[0][-1]

        compromised_nodes = set(idx for idx, value in enumerate(obs["compromised"]) if value)
        next_node = shortest_path_next(obs["adjacency"], current, target, compromised_nodes)
        if next_node is not None:
            action = ATTACK_EXPLOIT_OFFSET + n_nodes + next_node
            if valid[action]:
                return action
        return ATTACK_SCAN if valid[ATTACK_SCAN] else ATTACK_WAIT


class NoopDefender(BasePolicy):
    name = "noop"

    def act(self, obs):
        return DEFEND_NOOP


class RandomDefender(BasePolicy):
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs):
        choices = valid_indices(obs["valid_defender_actions"])
        return self.rng.choice(choices)


class PatchHighValueDefender(BasePolicy):
    name = "patch_high_value"

    def act(self, obs):
        n_nodes = len(obs["adjacency"])
        valid = obs["valid_defender_actions"]
        target = obs["target_node"]
        candidates = [target] + list(obs["adjacency"][target])
        candidates = sorted(set(candidates), key=lambda node: obs["values"][node], reverse=True)
        for node in candidates:
            action = 1 + node
            if valid[action] and not obs["patched"][node]:
                return action

        current = obs["current_node"]
        monitor_action = 1 + 2 * n_nodes + current
        if valid[monitor_action]:
            return monitor_action
        return DEFEND_NOOP


class DecoyFrontierDefender(BasePolicy):
    name = "decoy_frontier"

    def act(self, obs):
        n_nodes = len(obs["adjacency"])
        valid = obs["valid_defender_actions"]
        target = obs["target_node"]
        frontier = []
        for node, discovered in enumerate(obs["discovered"]):
            if not discovered or obs["compromised"][node]:
                continue
            if node in (obs["entry_node"], target):
                continue
            frontier.append((obs["dist_to_target"][node], obs["known_vuln"][node], node))
        frontier.sort()
        for _, _, node in frontier:
            action = 1 + n_nodes + node
            if valid[action] and not obs["decoys"][node]:
                return action

        current = obs["current_node"]
        monitor_action = 1 + 2 * n_nodes + current
        if valid[monitor_action]:
            return monitor_action
        return DEFEND_NOOP


class AdaptiveDefender(BasePolicy):
    name = "adaptive"

    def act(self, obs):
        n_nodes = len(obs["adjacency"])
        valid = obs["valid_defender_actions"]
        current = obs["current_node"]
        if obs["detection"] >= 3.0:
            monitor_action = 1 + 2 * n_nodes + current
            if valid[monitor_action]:
                return monitor_action

        target = obs["target_node"]
        for node in [target] + list(obs["adjacency"][target]):
            patch_action = 1 + node
            if valid[patch_action] and not obs["patched"][node]:
                return patch_action

        for node, discovered in enumerate(obs["discovered"]):
            if discovered and not obs["compromised"][node] and node not in (obs["entry_node"], target):
                decoy_action = 1 + n_nodes + node
                if valid[decoy_action] and not obs["decoys"][node]:
                    return decoy_action
        return DEFEND_NOOP


def default_attackers():
    return [RandomAttacker(seed=11), GreedyAttacker(), TargetedAttacker(), StealthAttacker()]


def default_defenders():
    return [
        NoopDefender(),
        RandomDefender(seed=17),
        PatchHighValueDefender(),
        DecoyFrontierDefender(),
        AdaptiveDefender(),
    ]
