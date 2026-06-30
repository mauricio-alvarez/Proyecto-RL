import argparse
import json

import numpy as np


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


def summarize_shapes(data, prefix):
    return {
        key: list(data[key].shape)
        for key in sorted(data.files)
        if key.startswith(prefix)
    }


def count_values(array):
    values, counts = np.unique(array, return_counts=True)
    return {str(value.item()): int(count.item()) for value, count in zip(values, counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    manifest = data["manifest"].item()
    summary = {
        "dataset": args.dataset,
        "schema_version": manifest.get("schema_version"),
        "env": manifest.get("env"),
        "hider_policy": manifest.get("hider_policy"),
        "seeker_policy": manifest.get("seeker_policy"),
        "episodes_collected": int(manifest.get("episodes_collected", 0)),
        "samples": int(manifest.get("samples", len(data["role"]))),
        "discard_count": int(manifest.get("discard_count", 0)),
        "role_counts": count_values(data["role"]),
        "winner_counts": count_values(data["episode_winner"]),
        "policy_id_counts": count_values(data["policy_id"]),
        "obs_shapes": summarize_shapes(data, "obs_"),
        "next_obs_shapes": summarize_shapes(data, "next_obs_"),
        "action_shapes": summarize_shapes(data, "action_"),
        "episode_summaries": make_serializable(data["episode_summaries"]),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("dataset:", summary["dataset"])
    print("schema_version:", summary["schema_version"])
    print("episodes_collected:", summary["episodes_collected"])
    print("samples:", summary["samples"])
    print("discard_count:", summary["discard_count"])
    print("role_counts:", summary["role_counts"])
    print("winner_counts:", summary["winner_counts"])
    print("policy_id_counts:", summary["policy_id_counts"])
    print("obs_shapes:", json.dumps(summary["obs_shapes"], sort_keys=True))
    print("action_shapes:", json.dumps(summary["action_shapes"], sort_keys=True))
    print("first_episode_summary:", json.dumps(summary["episode_summaries"][0], sort_keys=True))


if __name__ == "__main__":
    main()
