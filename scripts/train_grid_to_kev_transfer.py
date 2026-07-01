import argparse
import csv
import json
import math
import os
import random
import sys
from collections import deque

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cyber_rl.benchmark import default_attackers, default_defenders, summarize_results
from cyber_rl.env import CyberHideSeekEnv
from cyber_rl.kev import kev_families, load_kev_catalog, make_kev_scenario
from cyber_rl.policies import AdaptiveDefender, DecoyFrontierDefender, TargetedAttacker


CONCEPT_DIM = 32
GRID_ACTIONS = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def grid_obstacles(kind, size):
    obstacles = set()
    if kind == "empty":
        return obstacles
    if kind == "wall":
        col = size // 2
        gaps = {1, size - 2}
        for row in range(1, size - 1):
            if row not in gaps:
                obstacles.add((row, col))
        for col2 in range(2, size - 2):
            obstacles.add((size // 2, col2))
        obstacles.discard((size // 2, 1))
        obstacles.discard((size // 2, size - 2))
        return obstacles
    raise ValueError("unknown grid kind: {}".format(kind))


def in_bounds(pos, size):
    row, col = pos
    return 0 <= row < size and 0 <= col < size


def move_pos(pos, action, size, obstacles):
    deltas = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    dr, dc = deltas[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not in_bounds(nxt, size) or nxt in obstacles:
        return pos
    return nxt


def line_of_sight(a, b, obstacles):
    if a[0] == b[0]:
        row = a[0]
        low, high = sorted([a[1], b[1]])
        for col in range(low + 1, high):
            if (row, col) in obstacles:
                return False
        return True
    if a[1] == b[1]:
        col = a[1]
        low, high = sorted([a[0], b[0]])
        for row in range(low + 1, high):
            if (row, col) in obstacles:
                return False
        return True
    return False


def shortest_path_action(start, goal, size, obstacles):
    if start == goal:
        return 0
    queue = deque([start])
    prev = {start: None}
    prev_action = {}
    for action in range(1, GRID_ACTIONS):
        nxt = move_pos(start, action, size, obstacles)
        if nxt != start and nxt not in prev:
            prev[nxt] = start
            prev_action[nxt] = action
            queue.append(nxt)
    while queue:
        pos = queue.popleft()
        if pos == goal:
            cur = pos
            while prev[cur] != start:
                cur = prev[cur]
            return prev_action[cur]
        for action in range(1, GRID_ACTIONS):
            nxt = move_pos(pos, action, size, obstacles)
            if nxt != pos and nxt not in prev:
                prev[nxt] = pos
                prev_action[nxt] = action
                queue.append(nxt)
    return 0


def nearest_cover_distance(pos, size, obstacles):
    best = size * 2
    for row in range(size):
        for col in range(size):
            cell = (row, col)
            if cell in obstacles:
                continue
            cover = sum(1 for action in range(1, GRID_ACTIONS) if move_pos(cell, action, size, obstacles) == cell)
            cover += sum(1 for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (cell[0] + delta[0], cell[1] + delta[1]) in obstacles)
            if cover > 0:
                best = min(best, abs(pos[0] - row) + abs(pos[1] - col))
    return best


def ray_distance(pos, action, size, obstacles):
    distance = 0
    cur = pos
    while True:
        nxt = move_pos(cur, action, size, obstacles)
        if nxt == cur:
            return distance
        distance += 1
        cur = nxt


def adjacent_cover(pos, size, obstacles):
    count = 0
    for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nxt = (pos[0] + delta[0], pos[1] + delta[1])
        if not in_bounds(nxt, size) or nxt in obstacles:
            count += 1
    return count


def grid_concepts(role, seeker, hider, size, obstacles, step, max_steps):
    self_pos = seeker if role == "seeker" else hider
    other_pos = hider if role == "seeker" else seeker
    dx = float(other_pos[1] - self_pos[1]) / float(size - 1)
    dy = float(other_pos[0] - self_pos[0]) / float(size - 1)
    dist = abs(other_pos[0] - self_pos[0]) + abs(other_pos[1] - self_pos[1])
    los = line_of_sight(seeker, hider, obstacles)
    features = np.zeros(CONCEPT_DIM, dtype=np.float32)
    features[0] = 1.0
    features[1] = dx
    features[2] = dy
    features[3] = dist / float(2 * (size - 1))
    features[4] = 1.0 if los else 0.0
    features[5] = float(step) / float(max_steps)
    features[6] = float(size - 1 - step % size) / float(size)
    for idx, action in enumerate(range(1, GRID_ACTIONS)):
        features[7 + idx] = ray_distance(self_pos, action, size, obstacles) / float(size)
    features[11] = adjacent_cover(self_pos, size, obstacles) / 4.0
    features[12] = adjacent_cover(other_pos, size, obstacles) / 4.0
    features[13] = nearest_cover_distance(self_pos, size, obstacles) / float(2 * size)
    features[14] = nearest_cover_distance(other_pos, size, obstacles) / float(2 * size)
    features[15] = 1.0 if role == "seeker" else 0.0
    features[16] = 1.0 if role == "hider" else 0.0
    features[17] = float(self_pos[0]) / float(size - 1)
    features[18] = float(self_pos[1]) / float(size - 1)
    features[19] = float(other_pos[0]) / float(size - 1)
    features[20] = float(other_pos[1]) / float(size - 1)
    features[21] = len(obstacles) / float(size * size)
    features[22] = 1.0 if dist <= 1 else 0.0
    features[23] = 1.0 if dist <= 2 else 0.0
    features[24] = 1.0 if los and dist <= 3 else 0.0
    features[25] = max(0.0, 1.0 - dist / float(2 * (size - 1)))
    # Remaining entries are intentionally shared with cyber concepts as generic risk/progress slots.
    return features


def hider_expert_action(seeker, hider, size, obstacles, step, max_steps):
    best_action = 0
    best_score = -1e9
    for action in range(GRID_ACTIONS):
        nxt = move_pos(hider, action, size, obstacles)
        dist = abs(seeker[0] - nxt[0]) + abs(seeker[1] - nxt[1])
        los = line_of_sight(seeker, nxt, obstacles)
        cover = adjacent_cover(nxt, size, obstacles)
        cover_dist = nearest_cover_distance(nxt, size, obstacles)
        score = 2.0 * dist - 4.0 * float(los) + 1.5 * cover - 0.4 * cover_dist
        if nxt == seeker:
            score -= 100.0
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def seeker_expert_action(seeker, hider, size, obstacles, step, max_steps):
    if seeker == hider:
        return 0
    # The training world is fully observable so the seeker can learn the core pursuit
    # concept before we translate that representation into cyber graph pursuit.
    return shortest_path_action(seeker, hider, size, obstacles)


def sample_free_cell(rng, size, obstacles):
    while True:
        pos = (rng.randrange(size), rng.randrange(size))
        if pos not in obstacles:
            return pos


def generate_grid_dataset(samples, seed, role, size=7, max_steps=18):
    rng = random.Random(seed)
    xs = []
    ys = []
    kinds = ["empty", "wall"]
    for idx in range(samples):
        kind = kinds[idx % len(kinds)]
        obstacles = grid_obstacles(kind, size)
        seeker = sample_free_cell(rng, size, obstacles)
        hider = sample_free_cell(rng, size, obstacles)
        for _ in range(rng.randrange(0, max_steps)):
            h_action = hider_expert_action(seeker, hider, size, obstacles, 0, max_steps)
            s_action = seeker_expert_action(seeker, hider, size, obstacles, 0, max_steps)
            hider = move_pos(hider, h_action, size, obstacles)
            seeker = move_pos(seeker, s_action if rng.random() < 0.7 else rng.randrange(GRID_ACTIONS), size, obstacles)
        step = rng.randrange(max_steps)
        xs.append(grid_concepts(role, seeker, hider, size, obstacles, step, max_steps))
        if role == "seeker":
            ys.append(seeker_expert_action(seeker, hider, size, obstacles, step, max_steps))
        else:
            ys.append(hider_expert_action(seeker, hider, size, obstacles, step, max_steps))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


class ConceptPolicyNet(nn.Module):
    def __init__(self, action_dim, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(CONCEPT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden // 2, action_dim)

    def forward(self, x):
        return self.head(self.encoder(x))


def train_classifier(model, x_train, y_train, x_val, y_val, device, epochs, batch_size, lr):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    history = []
    n = x_train.shape[0]
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            train_acc = (model(x_train).argmax(dim=1) == y_train).float().mean().item()
            val_acc = (model(x_val_t).argmax(dim=1) == y_val_t).float().mean().item()
        history.append({
            "epoch": epoch,
            "loss": round(float(np.mean(losses)), 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
        })
    return history


def split_dataset(x, y, seed, frac=0.8):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    cut = int(len(idx) * frac)
    return x[idx[:cut]], y[idx[:cut]], x[idx[cut:]], y[idx[cut:]]


def evaluate_grid_model(model, role, kind, device, episodes=300, seed=9000, size=7, max_steps=18, opponent="expert"):
    rng = random.Random(seed)
    obstacles = grid_obstacles(kind, size)
    caught = 0
    total_steps = []
    final_distances = []
    cover_scores = []
    model.eval()
    for _ in range(episodes):
        seeker = sample_free_cell(rng, size, obstacles)
        hider = sample_free_cell(rng, size, obstacles)
        if seeker == hider:
            hider = sample_free_cell(rng, size, obstacles)
        done = False
        for step in range(max_steps):
            if role == "seeker":
                features = torch.from_numpy(grid_concepts("seeker", seeker, hider, size, obstacles, step, max_steps)).unsqueeze(0).to(device)
                with torch.no_grad():
                    seeker_action = int(model(features).argmax(dim=1).item())
                if opponent == "random":
                    hider_action = rng.randrange(GRID_ACTIONS)
                else:
                    hider_action = hider_expert_action(seeker, hider, size, obstacles, step, max_steps)
            else:
                if opponent == "random":
                    seeker_action = rng.randrange(GRID_ACTIONS)
                else:
                    seeker_action = seeker_expert_action(seeker, hider, size, obstacles, step, max_steps)
                features = torch.from_numpy(grid_concepts("hider", seeker, hider, size, obstacles, step, max_steps)).unsqueeze(0).to(device)
                with torch.no_grad():
                    hider_action = int(model(features).argmax(dim=1).item())
            seeker = move_pos(seeker, seeker_action, size, obstacles)
            hider = move_pos(hider, hider_action, size, obstacles)
            if seeker == hider:
                caught += 1
                total_steps.append(step + 1)
                done = True
                break
        if not done:
            total_steps.append(max_steps)
        final_distances.append(abs(seeker[0] - hider[0]) + abs(seeker[1] - hider[1]))
        cover_scores.append(adjacent_cover(hider, size, obstacles))
    return {
        "role": role,
        "environment": kind,
        "opponent": opponent,
        "episodes": episodes,
        "catch_rate": round(caught / float(episodes), 4),
        "survival_rate": round(1.0 - caught / float(episodes), 4),
        "mean_steps": round(float(np.mean(total_steps)), 4),
        "mean_final_distance": round(float(np.mean(final_distances)), 4),
        "mean_hider_cover": round(float(np.mean(cover_scores)), 4),
    }


def cyber_concepts(obs, role):
    n = float(max(len(obs["adjacency"]), 1))
    current = obs["current_node"]
    target = obs["target_node"]
    current_dist = min(obs["dist_to_target"][current], len(obs["adjacency"])) / n
    compromised = sum(1 for value in obs["compromised"] if value) / n
    discovered = sum(1 for value in obs["discovered"] if value) / n
    patched = sum(1 for value in obs["patched"] if value) / n
    decoys = sum(1 for value in obs["decoys"] if value) / n
    detection = min(float(obs["detection"]) / 6.0, 1.5)
    steps = float(obs["steps_remaining"]) / 24.0
    known_vulns = [value for value in obs["known_vuln"] if value >= 0]
    frontier = [idx for idx, discovered_flag in enumerate(obs["discovered"]) if discovered_flag and not obs["compromised"][idx]]
    features = np.zeros(CONCEPT_DIM, dtype=np.float32)
    features[0] = 1.0
    features[1] = -current_dist
    features[2] = compromised
    features[3] = discovered
    features[4] = detection
    features[5] = 1.0 - steps
    features[6] = steps
    features[7] = min(len(frontier) / n, 1.0)
    features[8] = np.mean(known_vulns) / 5.0 if known_vulns else 0.0
    features[9] = max(known_vulns) / 5.0 if known_vulns else 0.0
    features[10] = obs["values"][target] / 10.0
    features[11] = patched
    features[12] = decoys
    features[13] = obs["patch_budget_left"] / 3.0
    features[14] = obs["decoy_budget_left"] / 2.0
    features[15] = 1.0 if role == "attacker" else 0.0
    features[16] = 1.0 if role == "defender" else 0.0
    features[17] = float(current) / n
    features[18] = float(target) / n
    features[19] = min(len(obs["adjacency"][current]) / n, 1.0)
    features[20] = 1.0 if obs["active_monitor"] == current else 0.0
    features[21] = 1.0 if obs["valid_attacker_actions"][2] else 0.0
    features[22] = min(current_dist, 1.0)
    features[23] = max(0.0, 1.0 - current_dist)
    features[24] = 1.0 if detection > 0.7 else 0.0
    features[25] = 1.0 if len(frontier) > 0 else 0.0
    return features


def collect_cyber_demos(catalog, families, role, samples, seed, n_nodes, max_steps):
    rng = random.Random(seed)
    xs = []
    ys = []
    if role == "attacker":
        expert = TargetedAttacker()
    else:
        expert = DecoyFrontierDefender()
    defender = AdaptiveDefender()
    attacker = TargetedAttacker()
    while len(xs) < samples:
        family = families[len(xs) % len(families)]
        scenario = make_kev_scenario(catalog, family, seed * 100000 + len(xs), n_nodes=n_nodes, max_steps=max_steps)
        env = CyberHideSeekEnv(scenario, seed=seed + len(xs))
        obs = env.reset(seed=seed + len(xs))
        attacker.reset()
        defender.reset()
        expert.reset()
        done = False
        while not done and len(xs) < samples:
            if role == "attacker":
                label = expert.act(obs)
                xs.append(cyber_concepts(obs, "attacker"))
                ys.append(label)
                defender_action = defender.act(obs)
                attacker_action = label if rng.random() < 0.75 else rng.choice([idx for idx, flag in enumerate(obs["valid_attacker_actions"]) if flag])
            else:
                label = expert.act(obs)
                xs.append(cyber_concepts(obs, "defender"))
                ys.append(label)
                attacker_action = attacker.act(obs)
                defender_action = label if rng.random() < 0.75 else rng.choice([idx for idx, flag in enumerate(obs["valid_defender_actions"]) if flag])
            obs, _, done, _ = env.step(attacker_action, defender_action)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def init_transfer_model(grid_model, action_dim):
    model = ConceptPolicyNet(action_dim=action_dim)
    model.encoder.load_state_dict(grid_model.encoder.state_dict())
    return model


class NeuralPolicyWrapper(object):
    def __init__(self, model, role, device, name):
        self.model = model.to(device)
        self.model.eval()
        self.role = role
        self.device = device
        self.name = name

    def reset(self):
        return None

    def act(self, obs):
        x = torch.from_numpy(cyber_concepts(obs, self.role)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x).squeeze(0).detach().cpu().numpy()
        if self.role == "attacker":
            valid = obs["valid_attacker_actions"]
        else:
            valid = obs["valid_defender_actions"]
        masked = np.asarray(logits, dtype=np.float64)
        for idx, flag in enumerate(valid):
            if not flag:
                masked[idx] = -1e9
        return int(np.argmax(masked))


def run_neural_episode(scenario, attacker, defender, seed):
    env = CyberHideSeekEnv(scenario, seed=seed)
    obs = env.reset(seed=seed)
    attacker.reset()
    defender.reset()
    done = False
    while not done:
        defender_action = defender.act(obs)
        attacker_action = attacker.act(obs)
        obs, _, done, info = env.step(attacker_action, defender_action)
    return info


def evaluate_cyber_models(catalog, families, attackers, defenders, episodes_per_family, seed, n_nodes, max_steps):
    rows = []
    pair_summaries = []
    for attacker in attackers:
        for defender in defenders:
            records = []
            for family_idx, family in enumerate(families):
                for episode_idx in range(episodes_per_family):
                    scenario_seed = seed + family_idx * 10000 + episode_idx
                    scenario = make_kev_scenario(catalog, family, scenario_seed, n_nodes=n_nodes, max_steps=max_steps)
                    info = run_neural_episode(scenario, attacker, defender, seed=scenario_seed + 91)
                    row = {
                        "attacker": attacker.name,
                        "defender": defender.name,
                        "family": family,
                        "episode": episode_idx,
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
                    records.append(row)
            summary = summarize_results(records)
            summary["attacker"] = attacker.name
            summary["defender"] = defender.name
            pair_summaries.append(summary)
    aggregate = []
    for attacker_name in sorted(set(row["attacker"] for row in rows)):
        records = [row for row in rows if row["attacker"] == attacker_name]
        summary = summarize_results(records)
        summary["attacker"] = attacker_name
        aggregate.append(summary)
    defender_aggregate = []
    for defender_name in sorted(set(row["defender"] for row in rows)):
        records = [row for row in rows if row["defender"] == defender_name]
        summary = summarize_results(records)
        summary["defender"] = defender_name
        defender_aggregate.append(summary)
    return {
        "rows": rows,
        "pair_summaries": pair_summaries,
        "aggregate_by_attacker": aggregate,
        "aggregate_by_defender": defender_aggregate,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, payload):
    lines = []
    lines.append("# Grid-To-KEV Transfer Report")
    lines.append("")
    lines.append("## Device")
    lines.append("")
    lines.append("- device: `{}`".format(payload["device"]))
    lines.append("- cuda_available: `{}`".format(payload["cuda_available"]))
    lines.append("")
    lines.append("## Grid Training")
    lines.append("")
    for role in ["seeker", "hider"]:
        final = payload["grid_training"][role]["history"][-1]
        lines.append("- {} validation accuracy: {:.3f}".format(role, final["val_acc"]))
    lines.append("")
    lines.append("## Grid Concept Evaluation")
    lines.append("")
    lines.append("| role | environment | opponent | catch | survival | final distance | hider cover |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in payload["grid_evaluation"]:
        lines.append("| {role} | {environment} | {opponent} | {catch_rate:.3f} | {survival_rate:.3f} | {mean_final_distance:.2f} | {mean_hider_cover:.2f} |".format(**row))
    lines.append("")
    lines.append("## Cyber Transfer Evaluation")
    lines.append("")
    lines.append("| attacker | success | caught | timeout | return |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in sorted(payload["cyber_evaluation"]["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        lines.append("| {attacker} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} |".format(**row))
    lines.append("")
    lines.append("## Defender Transfer Evaluation")
    lines.append("")
    lines.append("| defender | attacker success | caught | timeout | attacker return |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in sorted(payload["cyber_evaluation"]["aggregate_by_defender"], key=lambda item: item["attacker_success_rate"]):
        lines.append("| {defender} | {attacker_success_rate:.3f} | {caught_rate:.3f} | {timeout_rate:.3f} | {mean_attacker_return:.2f} |".format(**row))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The grid seeker/hider models are trained on two simple game environments, then their encoder weights initialize cyber attacker/defender policies. The transferred cyber policies are fine-tuned on KEV-derived demonstrations and evaluated against principle baselines and scratch neural policies.")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kev-json", default="/workspace/data/raw/known_exploited_vulnerabilities.json")
    parser.add_argument("--out-dir", default="/workspace/runs/grid_to_kev_transfer_v1")
    parser.add_argument("--grid-samples", type=int, default=60000)
    parser.add_argument("--cyber-samples", type=int, default=24000)
    parser.add_argument("--grid-epochs", type=int, default=20)
    parser.add_argument("--cyber-epochs", type=int, default=18)
    parser.add_argument("--episodes-per-family", type=int, default=35)
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=24)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    grid_training = {}
    grid_models = {}
    for role in ["seeker", "hider"]:
        x, y = generate_grid_dataset(args.grid_samples, args.seed + (1 if role == "seeker" else 2), role=role)
        x_train, y_train, x_val, y_val = split_dataset(x, y, args.seed + 10)
        model = ConceptPolicyNet(action_dim=GRID_ACTIONS)
        history = train_classifier(model, x_train, y_train, x_val, y_val, device, args.grid_epochs, 1024, 1e-3)
        grid_training[role] = {
            "samples": int(len(x)),
            "history": history,
            "final_val_acc": history[-1]["val_acc"],
        }
        grid_models[role] = model
        torch.save({
            "role": role,
            "concept_dim": CONCEPT_DIM,
            "action_dim": GRID_ACTIONS,
            "model_state": model.state_dict(),
            "history": history,
        }, os.path.join(args.out_dir, "grid_{}.pt".format(role)))
        print("grid_role={} final_val_acc={:.4f}".format(role, history[-1]["val_acc"]))

    grid_eval = []
    for role in ["seeker", "hider"]:
        for kind in ["empty", "wall"]:
            opponents = ["random", "expert"] if role == "seeker" else ["random", "expert"]
            for opponent in opponents:
                grid_eval.append(evaluate_grid_model(grid_models[role], role, kind, device, seed=args.seed + 99, opponent=opponent))
            print("grid_eval", grid_eval[-1])

    catalog = load_kev_catalog(args.kev_json)
    families = kev_families()
    cyber_training = {}
    cyber_models = {}
    action_dims = {"attacker": 3 + 2 * args.n_nodes, "defender": 1 + 3 * args.n_nodes}
    role_map = {"attacker": "seeker", "defender": "hider"}

    for role in ["attacker", "defender"]:
        x, y = collect_cyber_demos(catalog, families, role, args.cyber_samples, args.seed + (20 if role == "attacker" else 30), args.n_nodes, args.max_steps)
        x_train, y_train, x_val, y_val = split_dataset(x, y, args.seed + 40)

        transfer_model = init_transfer_model(grid_models[role_map[role]], action_dims[role])
        transfer_history = train_classifier(transfer_model, x_train, y_train, x_val, y_val, device, args.cyber_epochs, 1024, 7e-4)
        scratch_model = ConceptPolicyNet(action_dim=action_dims[role])
        scratch_history = train_classifier(scratch_model, x_train, y_train, x_val, y_val, device, args.cyber_epochs, 1024, 7e-4)

        cyber_training[role] = {
            "samples": int(len(x)),
            "transfer_history": transfer_history,
            "scratch_history": scratch_history,
            "transfer_final_val_acc": transfer_history[-1]["val_acc"],
            "scratch_final_val_acc": scratch_history[-1]["val_acc"],
        }
        cyber_models["transfer_{}".format(role)] = transfer_model
        cyber_models["scratch_{}".format(role)] = scratch_model
        for name, model in [
            ("transfer_{}".format(role), transfer_model),
            ("scratch_{}".format(role), scratch_model),
        ]:
            torch.save({
                "role": role,
                "concept_dim": CONCEPT_DIM,
                "action_dim": action_dims[role],
                "model_state": model.state_dict(),
                "training": cyber_training[role],
            }, os.path.join(args.out_dir, "{}.pt".format(name)))
        print(
            "cyber_role={} transfer_val={:.4f} scratch_val={:.4f}".format(
                role,
                transfer_history[-1]["val_acc"],
                scratch_history[-1]["val_acc"],
            )
        )

    transfer_attacker = NeuralPolicyWrapper(cyber_models["transfer_attacker"], "attacker", device, "transfer_attacker")
    scratch_attacker = NeuralPolicyWrapper(cyber_models["scratch_attacker"], "attacker", device, "scratch_attacker")
    transfer_defender = NeuralPolicyWrapper(cyber_models["transfer_defender"], "defender", device, "transfer_defender")
    scratch_defender = NeuralPolicyWrapper(cyber_models["scratch_defender"], "defender", device, "scratch_defender")

    attackers = [TargetedAttacker(), transfer_attacker, scratch_attacker]
    defenders = [DecoyFrontierDefender(), AdaptiveDefender(), transfer_defender, scratch_defender]
    cyber_eval = evaluate_cyber_models(
        catalog,
        families,
        attackers,
        defenders,
        args.episodes_per_family,
        args.seed + 500,
        args.n_nodes,
        args.max_steps,
    )

    payload = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "grid_training": grid_training,
        "grid_evaluation": grid_eval,
        "cyber_training": cyber_training,
        "cyber_evaluation": cyber_eval,
        "kev_json": args.kev_json,
        "families": families,
        "n_nodes": args.n_nodes,
        "max_steps": args.max_steps,
    }
    with open(os.path.join(args.out_dir, "transfer_summary.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(os.path.join(args.out_dir, "cyber_transfer_episodes.csv"), cyber_eval["rows"])
    write_csv(os.path.join(args.out_dir, "cyber_transfer_pair_summaries.csv"), cyber_eval["pair_summaries"])
    write_report(os.path.join(args.out_dir, "transfer_report.md"), payload)

    print("saved_dir:", args.out_dir)
    print("grid_seeker_val:", grid_training["seeker"]["final_val_acc"])
    print("grid_hider_val:", grid_training["hider"]["final_val_acc"])
    print("cyber_attacker_transfer_val:", cyber_training["attacker"]["transfer_final_val_acc"])
    print("cyber_attacker_scratch_val:", cyber_training["attacker"]["scratch_final_val_acc"])
    print("cyber_defender_transfer_val:", cyber_training["defender"]["transfer_final_val_acc"])
    print("cyber_defender_scratch_val:", cyber_training["defender"]["scratch_final_val_acc"])
    for row in sorted(cyber_eval["aggregate_by_attacker"], key=lambda item: item["attacker_success_rate"], reverse=True):
        print(
            "attacker={attacker} success={attacker_success_rate:.3f} caught={caught_rate:.3f} "
            "timeout={timeout_rate:.3f} return={mean_attacker_return:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
