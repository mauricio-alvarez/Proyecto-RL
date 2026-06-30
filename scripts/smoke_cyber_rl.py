import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cyber_rl.benchmark import run_benchmark_suite
from cyber_rl.env import CyberHideSeekEnv, make_scenario
from cyber_rl.policies import GreedyAttacker, PatchHighValueDefender, RandomAttacker, RandomDefender


def run_episode(attacker, defender, family, seed):
    scenario = make_scenario(family, seed, n_nodes=8, max_steps=24)
    env = CyberHideSeekEnv(scenario, seed=seed)
    obs = env.reset(seed=seed)
    attacker.reset()
    defender.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(attacker.act(obs), defender.act(obs))
    assert info["outcome"] in ("attacker_success", "caught", "timeout")
    assert info["steps"] <= scenario.max_steps
    assert info["compromised_count"] >= 1
    return info


def main():
    random_info = run_episode(RandomAttacker(seed=1), RandomDefender(seed=2), "random", 123)
    greedy_info = run_episode(GreedyAttacker(), PatchHighValueDefender(), "branching", 124)
    suite = run_benchmark_suite(episodes_per_family=2, seed=700, families=["chain", "random"])
    assert len(suite["rows"]) > 0
    assert len(suite["pair_summaries"]) > 0
    print("cyber_smoke_ok")
    print("random_outcome:", random_info["outcome"])
    print("greedy_outcome:", greedy_info["outcome"])
    print("benchmark_rows:", len(suite["rows"]))
    print("pair_summaries:", len(suite["pair_summaries"]))


if __name__ == "__main__":
    main()
