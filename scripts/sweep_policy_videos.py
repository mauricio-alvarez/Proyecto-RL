import argparse
import json
import os
import re
import subprocess
import sys
import time


DEFAULT_POLICIES = [
    "hide_and_seek_quadrant.npz",
    "hide_and_seek_full.npz",
    "hide_and_seek_quadrant_physics_exploits.npz",
    "hide_and_seek_policy_phases/a_chasing.npz",
    "hide_and_seek_policy_phases/b_forts.npz",
    "hide_and_seek_policy_phases/c_ramps.npz",
    "hide_and_seek_policy_phases/d_ramp_defense.npz",
    "hide_and_seek_policy_phases/e_box_surfing.npz",
]

QUADRANT_ENV = "/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet"
FULL_ENV = "/workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet"
QUADRANT_POLICIES = {
    "hide_and_seek_quadrant.npz",
    "hide_and_seek_quadrant_physics_exploits.npz",
}


def sanitize_name(policy_path):
    name = policy_path.replace("\\", "/").rstrip("/").split("/")[-1]
    if name.endswith(".npz"):
        name = name[:-4]
    parent = policy_path.replace("\\", "/").split("/")[-2:-1]
    if parent and parent[0] == "hide_and_seek_policy_phases":
        name = parent[0] + "_" + name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def select_env(policy_rel, env_arg, quadrant_env, full_env):
    if env_arg != "auto":
        return env_arg

    normalized = policy_rel.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    if basename in QUADRANT_POLICIES:
        return quadrant_env
    return full_env


def tail(text, max_chars=2400):
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_command(command):
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return proc.returncode, proc.stdout


def classify_failure(output):
    if "Error assigning weights of shape" in output:
        return "incompatible_observation_schema"
    if "AssertionError" in output and "load_variables" in output:
        return "incompatible_policy_weights"
    return "failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="auto",
        help="Use 'auto' to route each built-in policy to its compatible env.",
    )
    parser.add_argument("--quadrant-env", default=QUADRANT_ENV)
    parser.add_argument("--full-env", default=FULL_ENV)
    parser.add_argument(
        "--policy-root",
        default="/workspace/multi-agent-emergence-environments/examples",
    )
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out-dir", default="/workspace/runs/policy_sweep")
    parser.add_argument("--video-dir", default="/workspace/videos/policy_sweep")
    parser.add_argument("--summary-out", default="/workspace/runs/policy_sweep/summary.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    if os.path.dirname(args.summary_out):
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)

    policies = args.policy or DEFAULT_POLICIES
    results = []

    for policy_rel in policies:
        policy_path = policy_rel
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(args.policy_root, policy_rel)
        env_path = select_env(
            policy_rel=policy_rel,
            env_arg=args.env,
            quadrant_env=args.quadrant_env,
            full_env=args.full_env,
        )

        name = sanitize_name(policy_rel)
        rollout_path = os.path.join(args.out_dir, "{}.npz".format(name))
        video_path = os.path.join(args.video_dir, "{}_ep0.gif".format(name))

        print("=== policy: {} ===".format(policy_rel))
        print("env:", env_path)
        collect_cmd = [
            sys.executable,
            "/workspace/scripts/collect_policy_rollouts.py",
            "--env", env_path,
            "--policy", policy_path,
            "--episodes", str(args.episodes),
            "--steps", str(args.steps),
            "--seed", str(args.seed),
            "--out", rollout_path,
        ]
        collect_code, collect_output = run_command(collect_cmd)
        print(tail(collect_output))

        result = {
            "policy": policy_rel,
            "policy_path": policy_path,
            "env": env_path,
            "rollout_path": rollout_path,
            "video_path": video_path,
            "collect_returncode": collect_code,
            "created_unix": time.time(),
        }

        collect_failed = (
            collect_code != 0
            or "Error assigning weights of shape" in collect_output
            or not os.path.exists(rollout_path)
        )
        if collect_failed:
            result["status"] = classify_failure(collect_output)
            if collect_code == 0 and not os.path.exists(rollout_path):
                result["status"] = "incompatible_observation_schema"
            result["error_tail"] = tail(collect_output)
            print("status:", result["status"])
            results.append(result)
            continue

        render_cmd = [
            sys.executable,
            "/workspace/scripts/render_rollout_video.py",
            "--input", rollout_path,
            "--episode", "0",
            "--out", video_path,
            "--fps", str(args.fps),
        ]
        render_code, render_output = run_command(render_cmd)
        print(tail(render_output))

        result["render_returncode"] = render_code
        if render_code == 0:
            result["status"] = "video_created"
        else:
            result["status"] = "render_failed"
            result["error_tail"] = tail(render_output)
        results.append(result)
        print("status:", result["status"])

    summary = {
        "env": args.env,
        "quadrant_env": args.quadrant_env,
        "full_env": args.full_env,
        "episodes_per_policy": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "video_dir": args.video_dir,
        "out_dir": args.out_dir,
        "results": results,
        "video_count": sum(1 for result in results if result["status"] == "video_created"),
        "failed_count": sum(1 for result in results if result["status"] != "video_created"),
    }

    with open(args.summary_out, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print("summary:", args.summary_out)
    print("video_count:", summary["video_count"])
    print("failed_count:", summary["failed_count"])


if __name__ == "__main__":
    main()
