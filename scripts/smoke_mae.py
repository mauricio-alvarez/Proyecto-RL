import argparse
import numpy as np

from mujoco_worldgen.util.envs import load_env


def sample_action(env):
    action = env.action_space.sample()
    return {key: np.array(value) for key, value in action.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    env = load_env(
        args.env,
        core_dir="/workspace/multi-agent-emergence-environments",
        envs_dir="mae_envs/envs",
        xmls_dir="xmls",
    )

    print("action_space:", env.action_space)
    print("observation_keys:", sorted(env.reset().keys()))

    for episode in range(args.episodes):
        env.reset()
        total_reward = None
        for step in range(args.steps):
            obs, rew, done, info = env.step(sample_action(env))
            total_reward = rew if total_reward is None else total_reward + rew
            if done or info.get("discard_episode"):
                break
        print(
            "episode={} steps={} total_reward={} done={} discard={}".format(
                episode,
                step + 1,
                np.asarray(total_reward).round(3).tolist(),
                done,
                info.get("discard_episode"),
            )
        )


if __name__ == "__main__":
    main()
