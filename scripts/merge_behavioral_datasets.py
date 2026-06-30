import argparse
import json
import os
import time

import numpy as np


RESERVED_KEYS = set(["manifest", "episode_summaries", "episode_layouts"])


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


def load_manifest(data):
    if "manifest" not in data.files:
        return {}
    value = data["manifest"]
    return value.item() if value.shape == () else value.tolist()


def load_object_array(data, key):
    if key not in data.files:
        return []
    return [item for item in data[key].tolist()]


def offset_episode_summaries(summaries, episode_offset, source_path):
    output = []
    for summary in summaries:
        if isinstance(summary, dict):
            item = dict(summary)
            if "episode" in item:
                item["episode"] = int(item["episode"]) + episode_offset
            item["source_dataset"] = source_path
            output.append(item)
        else:
            output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    if len(args.datasets) < 2:
        raise ValueError("Provide at least two input datasets to merge")

    datasets = [np.load(path, allow_pickle=True) for path in args.datasets]
    common_keys = set(datasets[0].files) - RESERVED_KEYS
    for data in datasets[1:]:
        common_keys &= set(data.files) - RESERVED_KEYS
    common_keys = sorted(common_keys)
    if "episode" not in common_keys:
        raise ValueError("All datasets must contain an episode array")

    merged = {key: [] for key in common_keys}
    episode_summaries = []
    episode_layouts = []
    input_records = []
    next_episode_offset = 0

    for path, data in zip(args.datasets, datasets):
        episodes = data["episode"].astype(np.int32)
        unique_episodes = sorted(np.unique(episodes).astype(np.int32).tolist())
        if unique_episodes:
            local_min = int(unique_episodes[0])
            local_max = int(unique_episodes[-1])
        else:
            local_min = 0
            local_max = -1

        episode_offset = next_episode_offset - local_min
        adjusted_episodes = episodes + episode_offset

        for key in common_keys:
            if key == "episode":
                merged[key].append(adjusted_episodes.astype(data[key].dtype))
            else:
                merged[key].append(data[key])

        summaries = load_object_array(data, "episode_summaries")
        episode_summaries.extend(offset_episode_summaries(summaries, episode_offset, path))
        episode_layouts.extend(load_object_array(data, "episode_layouts"))

        manifest = load_manifest(data)
        sample_count = int(len(data["role"])) if "role" in data.files else int(len(episodes))
        input_records.append({
            "path": path,
            "samples": sample_count,
            "episodes": int(len(unique_episodes)),
            "episode_min": local_min,
            "episode_max": local_max,
            "episode_offset": int(episode_offset),
            "collector": manifest.get("collector"),
            "schema_version": manifest.get("schema_version"),
        })
        next_episode_offset = int(adjusted_episodes.max()) + 1 if len(adjusted_episodes) else next_episode_offset

    arrays = {key: np.concatenate(values, axis=0) for key, values in merged.items()}
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "schema_version": "behavioral_dataset_v1",
        "collector": "merge_behavioral_datasets.py",
        "created_unix": time.time(),
        "inputs": input_records,
        "samples": int(len(arrays["role"])) if "role" in arrays else int(len(arrays["episode"])),
        "episodes_collected": int(len(np.unique(arrays["episode"]))),
        "merged_keys": common_keys,
        "notes": [
            "Episode ids are offset so train/validation episode splits remain valid.",
            "Only keys common to every input dataset are retained.",
        ],
    }

    np.savez_compressed(
        args.out,
        manifest=np.array(manifest, dtype=object),
        episode_summaries=np.array(episode_summaries, dtype=object),
        episode_layouts=np.array(episode_layouts, dtype=object),
        **arrays
    )

    print("saved:", args.out)
    print("samples:", manifest["samples"])
    print("episodes_collected:", manifest["episodes_collected"])
    print("merged_keys:", len(common_keys))
    print("manifest_json:", json.dumps(make_serializable(manifest), sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
