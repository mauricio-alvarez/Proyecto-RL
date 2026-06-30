import random
from collections import deque


ATTACK_WAIT = 0
ATTACK_SCAN = 1
ATTACK_EXFILTRATE = 2
ATTACK_EXPLOIT_OFFSET = 3

DEFEND_NOOP = 0


def clamp(value, low, high):
    return max(low, min(high, value))


def bitmask(flags):
    value = 0
    for idx, flag in enumerate(flags):
        if flag:
            value |= 1 << idx
    return value


def shortest_distances(adjacency, target):
    distances = [999 for _ in adjacency]
    distances[target] = 0
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if distances[nxt] == 999:
                distances[nxt] = distances[node] + 1
                queue.append(nxt)
    return distances


class CyberScenario(object):
    def __init__(
        self,
        name,
        adjacency,
        entry_node,
        target_node,
        vulnerabilities,
        values,
        max_steps=24,
        detection_limit=6.0,
        patch_budget=3,
        decoy_budget=2,
        node_metadata=None,
        source=None,
    ):
        self.name = name
        self.adjacency = [sorted(list(neighbors)) for neighbors in adjacency]
        self.entry_node = int(entry_node)
        self.target_node = int(target_node)
        self.vulnerabilities = [int(value) for value in vulnerabilities]
        self.values = [float(value) for value in values]
        self.max_steps = int(max_steps)
        self.detection_limit = float(detection_limit)
        self.patch_budget = int(patch_budget)
        self.decoy_budget = int(decoy_budget)
        self.n_nodes = len(self.adjacency)
        self.dist_to_target = shortest_distances(self.adjacency, self.target_node)
        self.node_metadata = node_metadata or [{} for _ in range(self.n_nodes)]
        self.source = source or {}

    def to_dict(self):
        return {
            "name": self.name,
            "adjacency": self.adjacency,
            "entry_node": self.entry_node,
            "target_node": self.target_node,
            "vulnerabilities": self.vulnerabilities,
            "values": self.values,
            "max_steps": self.max_steps,
            "detection_limit": self.detection_limit,
            "patch_budget": self.patch_budget,
            "decoy_budget": self.decoy_budget,
            "node_metadata": self.node_metadata,
            "source": self.source,
        }


def _empty_graph(n_nodes):
    return [set() for _ in range(n_nodes)]


def _add_edge(graph, left, right):
    if left == right:
        return
    graph[left].add(right)
    graph[right].add(left)


def _finalize_graph(graph):
    return [sorted(list(neighbors)) for neighbors in graph]


def make_scenario(family, seed, n_nodes=8, max_steps=24):
    rng = random.Random(seed)
    n_nodes = int(n_nodes)
    graph = _empty_graph(n_nodes)

    if family == "chain":
        for idx in range(n_nodes - 1):
            _add_edge(graph, idx, idx + 1)
        for _ in range(max(1, n_nodes // 4)):
            left = rng.randrange(0, n_nodes - 2)
            _add_edge(graph, left, min(n_nodes - 1, left + 2))
    elif family == "branching":
        for idx in range(1, n_nodes):
            parent = (idx - 1) // 2
            _add_edge(graph, parent, idx)
        for _ in range(n_nodes // 3):
            _add_edge(graph, rng.randrange(n_nodes), rng.randrange(n_nodes))
    elif family == "dense":
        for idx in range(n_nodes - 1):
            _add_edge(graph, idx, idx + 1)
        for left in range(n_nodes):
            for right in range(left + 1, n_nodes):
                if rng.random() < 0.35:
                    _add_edge(graph, left, right)
    elif family == "decoy_heavy":
        for idx in range(1, n_nodes):
            if idx <= n_nodes // 2:
                _add_edge(graph, idx - 1, idx)
            else:
                _add_edge(graph, rng.randrange(0, n_nodes // 2), idx)
        for _ in range(n_nodes // 2):
            _add_edge(graph, rng.randrange(n_nodes), rng.randrange(n_nodes))
    elif family == "random":
        for idx in range(1, n_nodes):
            _add_edge(graph, idx, rng.randrange(idx))
        for _ in range(n_nodes):
            _add_edge(graph, rng.randrange(n_nodes), rng.randrange(n_nodes))
    else:
        raise ValueError("Unknown scenario family: {}".format(family))

    entry_node = 0
    target_node = n_nodes - 1
    vulnerabilities = [rng.randint(1, 5) for _ in range(n_nodes)]
    values = [rng.uniform(1.0, 5.0) for _ in range(n_nodes)]
    vulnerabilities[entry_node] = 1
    vulnerabilities[target_node] = max(vulnerabilities[target_node], 4)
    values[target_node] = 10.0

    if family == "decoy_heavy":
        for idx in range(n_nodes // 2, n_nodes - 1):
            vulnerabilities[idx] = max(vulnerabilities[idx], 4)
            values[idx] = rng.uniform(0.5, 2.0)

    return CyberScenario(
        name="{}_seed{}".format(family, seed),
        adjacency=_finalize_graph(graph),
        entry_node=entry_node,
        target_node=target_node,
        vulnerabilities=vulnerabilities,
        values=values,
        max_steps=max_steps,
        detection_limit=6.0,
        patch_budget=3,
        decoy_budget=2,
    )


class CyberHideSeekEnv(object):
    """Small zero-sum cyber hide-and-seek environment.

    Seeker: scans, exploits, moves, and exfiltrates from the target.
    Hider: patches services, deploys decoys, and monitors likely attacker nodes.
    """

    def __init__(self, scenario, seed=0):
        self.scenario = scenario
        self.n_nodes = scenario.n_nodes
        self.attack_action_count = ATTACK_EXPLOIT_OFFSET + 2 * self.n_nodes
        self.defend_action_count = 1 + 3 * self.n_nodes
        self.rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        n = self.n_nodes
        self.step_count = 0
        self.done = False
        self.outcome = None
        self.current_node = self.scenario.entry_node
        self.discovered = [False] * n
        self.compromised = [False] * n
        self.scanned = [False] * n
        self.known_vuln = [-1] * n
        self.patched = [False] * n
        self.decoys = [False] * n
        self.active_monitor = None
        self.patch_budget_left = self.scenario.patch_budget
        self.decoy_budget_left = self.scenario.decoy_budget
        self.detection = 0.0
        self.total_attacker_reward = 0.0
        self.event_counts = {
            "scans": 0,
            "exploit_successes": 0,
            "exploit_failures": 0,
            "moves": 0,
            "invalid_actions": 0,
            "patches": 0,
            "decoys": 0,
            "monitors": 0,
        }

        self.discovered[self.current_node] = True
        self.compromised[self.current_node] = True
        self.known_vuln[self.current_node] = self.scenario.vulnerabilities[self.current_node]
        return self.observation()

    def exploit_action(self, node):
        return ATTACK_EXPLOIT_OFFSET + int(node)

    def move_action(self, node):
        return ATTACK_EXPLOIT_OFFSET + self.n_nodes + int(node)

    def patch_action(self, node):
        return 1 + int(node)

    def decoy_action(self, node):
        return 1 + self.n_nodes + int(node)

    def monitor_action(self, node):
        return 1 + 2 * self.n_nodes + int(node)

    def valid_attacker_actions(self):
        valid = [False] * self.attack_action_count
        valid[ATTACK_WAIT] = True
        valid[ATTACK_SCAN] = True
        if self.current_node == self.scenario.target_node and self.compromised[self.current_node]:
            valid[ATTACK_EXFILTRATE] = True
        for node in self.scenario.adjacency[self.current_node]:
            if self.discovered[node] and not self.compromised[node]:
                valid[self.exploit_action(node)] = True
            if self.compromised[node]:
                valid[self.move_action(node)] = True
        return valid

    def valid_defender_actions(self):
        valid = [True] * self.defend_action_count
        if self.patch_budget_left <= 0:
            for node in range(self.n_nodes):
                valid[self.patch_action(node)] = False
        if self.decoy_budget_left <= 0:
            for node in range(self.n_nodes):
                valid[self.decoy_action(node)] = False
        return valid

    def observation(self):
        return {
            "scenario_name": self.scenario.name,
            "adjacency": self.scenario.adjacency,
            "current_node": self.current_node,
            "entry_node": self.scenario.entry_node,
            "target_node": self.scenario.target_node,
            "discovered": list(self.discovered),
            "compromised": list(self.compromised),
            "scanned": list(self.scanned),
            "known_vuln": list(self.known_vuln),
            "detection": self.detection,
            "steps_remaining": self.scenario.max_steps - self.step_count,
            "valid_attacker_actions": self.valid_attacker_actions(),
            "valid_defender_actions": self.valid_defender_actions(),
            "patch_budget_left": self.patch_budget_left,
            "decoy_budget_left": self.decoy_budget_left,
            "patched": list(self.patched),
            "decoys": list(self.decoys),
            "active_monitor": self.active_monitor,
            "dist_to_target": list(self.scenario.dist_to_target),
            "values": list(self.scenario.values),
        }

    def _apply_defender_action(self, action):
        if action == DEFEND_NOOP:
            self.active_monitor = None
            return
        if 1 <= action < 1 + self.n_nodes:
            node = action - 1
            if self.patch_budget_left > 0 and not self.patched[node]:
                self.patched[node] = True
                self.patch_budget_left -= 1
                self.event_counts["patches"] += 1
            return
        if 1 + self.n_nodes <= action < 1 + 2 * self.n_nodes:
            node = action - 1 - self.n_nodes
            if (
                self.decoy_budget_left > 0
                and node != self.scenario.entry_node
                and node != self.scenario.target_node
                and not self.decoys[node]
            ):
                self.decoys[node] = True
                self.decoy_budget_left -= 1
                self.event_counts["decoys"] += 1
            return
        if 1 + 2 * self.n_nodes <= action < 1 + 3 * self.n_nodes:
            self.active_monitor = action - 1 - 2 * self.n_nodes
            self.event_counts["monitors"] += 1

    def _scan(self):
        info_gain = 0
        self.scanned[self.current_node] = True
        for node in self.scenario.adjacency[self.current_node]:
            if not self.discovered[node]:
                info_gain += 1
            self.discovered[node] = True
            self.known_vuln[node] = self.scenario.vulnerabilities[node]
        self.event_counts["scans"] += 1
        self.detection += 0.35
        if self.active_monitor == self.current_node:
            self.detection += 0.9
        return -0.5 + 0.5 * info_gain

    def _exploit_success_probability(self, node):
        vuln = self.scenario.vulnerabilities[node]
        probability = 0.22 + 0.13 * vuln
        if self.patched[node]:
            probability -= 0.35
        if self.active_monitor == node:
            probability -= 0.05
        return clamp(probability, 0.05, 0.95)

    def _exploit(self, node):
        if node not in self.scenario.adjacency[self.current_node] or not self.discovered[node] or self.compromised[node]:
            self.event_counts["invalid_actions"] += 1
            self.detection += 0.6
            return -5.0
        if self.decoys[node]:
            self.done = True
            self.outcome = "caught"
            self.detection = self.scenario.detection_limit
            return -100.0

        success_probability = self._exploit_success_probability(node)
        if self.rng.random() < success_probability:
            self.compromised[node] = True
            self.discovered[node] = True
            self.known_vuln[node] = self.scenario.vulnerabilities[node]
            self.event_counts["exploit_successes"] += 1
            self.detection += 0.8
            if self.active_monitor == node:
                self.detection += 1.4
            if self.patched[node]:
                self.detection += 0.5
            progress = max(0, self.scenario.dist_to_target[self.current_node] - self.scenario.dist_to_target[node])
            return 6.0 + 0.25 * self.scenario.values[node] + 2.0 * progress

        self.event_counts["exploit_failures"] += 1
        self.detection += 1.2
        if self.active_monitor == node:
            self.detection += 1.2
        if self.patched[node]:
            self.detection += 0.6
        return -4.0

    def _move(self, node):
        if node not in self.scenario.adjacency[self.current_node] or not self.compromised[node]:
            self.event_counts["invalid_actions"] += 1
            self.detection += 0.4
            return -4.0
        self.current_node = node
        self.event_counts["moves"] += 1
        if self.active_monitor == node:
            self.detection += 0.7
        return -1.0

    def _exfiltrate(self):
        if self.current_node == self.scenario.target_node and self.compromised[self.current_node]:
            self.done = True
            self.outcome = "attacker_success"
            return 100.0
        self.detection += 1.5
        self.event_counts["invalid_actions"] += 1
        return -10.0

    def _apply_attacker_action(self, action):
        if action == ATTACK_WAIT:
            self.detection = max(0.0, self.detection - 0.4)
            return -1.0
        if action == ATTACK_SCAN:
            return self._scan()
        if action == ATTACK_EXFILTRATE:
            return self._exfiltrate()
        if ATTACK_EXPLOIT_OFFSET <= action < ATTACK_EXPLOIT_OFFSET + self.n_nodes:
            return self._exploit(action - ATTACK_EXPLOIT_OFFSET)
        if ATTACK_EXPLOIT_OFFSET + self.n_nodes <= action < ATTACK_EXPLOIT_OFFSET + 2 * self.n_nodes:
            return self._move(action - ATTACK_EXPLOIT_OFFSET - self.n_nodes)

        self.event_counts["invalid_actions"] += 1
        self.detection += 0.5
        return -5.0

    def step(self, attacker_action, defender_action):
        if self.done:
            return self.observation(), 0.0, True, self.info()

        self._apply_defender_action(int(defender_action))
        reward = self._apply_attacker_action(int(attacker_action))
        self.step_count += 1

        if not self.done and self.detection >= self.scenario.detection_limit:
            self.done = True
            self.outcome = "caught"
            reward -= 100.0

        if not self.done and self.step_count >= self.scenario.max_steps:
            self.done = True
            self.outcome = "timeout"
            reward -= 80.0

        self.total_attacker_reward += reward
        return self.observation(), reward, self.done, self.info()

    def info(self):
        compromised_count = sum(1 for value in self.compromised if value)
        return {
            "outcome": self.outcome,
            "attacker_success": self.outcome == "attacker_success",
            "defender_success": self.outcome in ("caught", "timeout"),
            "caught": self.outcome == "caught",
            "timeout": self.outcome == "timeout",
            "steps": self.step_count,
            "detection": round(self.detection, 4),
            "compromised_count": compromised_count,
            "compromised_mask": bitmask(self.compromised),
            "discovered_mask": bitmask(self.discovered),
            "attacker_return": round(self.total_attacker_reward, 4),
            "event_counts": dict(self.event_counts),
            "scenario": self.scenario.to_dict(),
        }
