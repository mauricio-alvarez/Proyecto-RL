import argparse
import json
import os
import re
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont


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


def load_font(size=14, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except IOError:
            pass
    return ImageFont.load_default()


def sanitize_name(policy_path):
    name = policy_path.replace("\\", "/").rstrip("/").split("/")[-1]
    if name.endswith(".npz"):
        name = name[:-4]
    parent = policy_path.replace("\\", "/").split("/")[-2:-1]
    if parent and parent[0] == "hide_and_seek_policy_phases":
        name = parent[0] + "_" + name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def display_name(policy_path):
    return sanitize_name(policy_path).replace("hide_and_seek_policy_phases_", "phase_")


def select_env(policy_rel, env_arg, quadrant_env, full_env):
    if env_arg != "auto":
        return env_arg
    basename = policy_rel.replace("\\", "/").rsplit("/", 1)[-1]
    if basename in QUADRANT_POLICIES:
        return quadrant_env
    return full_env


def short_env(env_path):
    name = os.path.basename(env_path)
    return name.replace(".jsonnet", "")


def run_command(command):
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return proc.returncode, proc.stdout


def tail(text, max_chars=2400):
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def load_eval_json(path):
    with open(path) as input_file:
        return json.load(input_file)


def metric(result, key, default=0.0):
    value = result.get(key, default)
    return default if value is None else value


def seeker_win_rate(result):
    episodes = max(int(result.get("episodes", 0)), 1)
    return float(result.get("seeker_wins", 0)) / episodes


def hider_win_rate(result):
    episodes = max(int(result.get("episodes", 0)), 1)
    return float(result.get("hider_wins", 0)) / episodes


def tie_rate(result):
    episodes = max(int(result.get("episodes", 0)), 1)
    return float(result.get("ties", 0)) / episodes


def draw_axes(draw, box, y_min, y_max, label, font):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(45, 45, 45), width=1)
    draw.text((x0, y0 - 22), label, fill=(20, 20, 20), font=font)
    for idx in range(5):
        frac = idx / 4.0
        y = y1 - frac * (y1 - y0)
        value = y_min + frac * (y_max - y_min)
        draw.line((x0, y, x1, y), fill=(228, 228, 228))
        draw.text((x0 - 58, y - 7), "{:.2f}".format(value), fill=(75, 75, 75), font=font)


def plot_points(draw, box, values, y_min, y_max, color, width=3):
    if not values:
        return
    x0, y0, x1, y1 = box
    if len(values) == 1:
        xs = [(x0 + x1) / 2.0]
    else:
        xs = [x0 + idx * (x1 - x0) / float(len(values) - 1) for idx in range(len(values))]
    denom = max(y_max - y_min, 1e-9)
    points = []
    for x, value in zip(xs, values):
        frac = (value - y_min) / denom
        y = y1 - frac * (y1 - y0)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=color, width=width)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def draw_policy_plot(result, plot_path):
    episodes = result.get("episodes_detail", [])
    name = display_name(result["policy_rel"])

    image = Image.new("RGB", (1100, 760), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    title_font = load_font(22, bold=True)
    label_font = load_font(14)
    small_font = load_font(12)

    draw.text((30, 24), name, fill=(20, 20, 20), font=title_font)
    draw.text(
        (30, 54),
        "env={}  episodes={}  seeker_wins={}  hider_wins={}  ties={}".format(
            short_env(result["env"]),
            result.get("episodes", 0),
            result.get("seeker_wins", 0),
            result.get("hider_wins", 0),
            result.get("ties", 0),
        ),
        fill=(55, 55, 55),
        font=label_font,
    )

    hider_returns = [float(ep["hider_mean_return"]) for ep in episodes]
    seeker_returns = [float(ep["seeker_mean_return"]) for ep in episodes]
    visible = [float(ep["visible_fraction"]) for ep in episodes]
    first_visible = [
        float(ep["first_visible_step"]) if ep["first_visible_step"] is not None else float(ep["steps"])
        for ep in episodes
    ]

    returns = hider_returns + seeker_returns
    y_min = min(returns) if returns else -1.0
    y_max = max(returns) if returns else 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = max((y_max - y_min) * 0.1, 1.0)
    y_min -= pad
    y_max += pad

    returns_box = (105, 125, 1040, 330)
    draw_axes(draw, returns_box, y_min, y_max, "Mean team return per episode", small_font)
    plot_points(draw, returns_box, hider_returns, y_min, y_max, (53, 150, 83))
    plot_points(draw, returns_box, seeker_returns, y_min, y_max, (195, 76, 65))
    draw.text((850, 105), "hider", fill=(53, 150, 83), font=label_font)
    draw.text((920, 105), "seeker", fill=(195, 76, 65), font=label_font)

    visible_box = (105, 405, 560, 620)
    draw_axes(draw, visible_box, 0.0, 1.0, "Visible fraction", small_font)
    plot_points(draw, visible_box, visible, 0.0, 1.0, (56, 98, 170))

    first_box = (650, 405, 1040, 620)
    max_step = max(first_visible) if first_visible else 1.0
    max_step = max(max_step, 1.0)
    draw_axes(draw, first_box, 0.0, max_step, "First visible step", small_font)
    plot_points(draw, first_box, first_visible, 0.0, max_step, (120, 80, 150))

    stats = [
        ("seeker win rate", seeker_win_rate(result)),
        ("hider win rate", hider_win_rate(result)),
        ("tie rate", tie_rate(result)),
        ("mean visible fraction", metric(result, "mean_visible_fraction")),
        ("mean seeker return", metric(result, "mean_seeker_return")),
    ]
    x = 105
    y = 670
    for label, value in stats:
        draw.text((x, y), "{}: {:.3f}".format(label, value), fill=(35, 35, 35), font=label_font)
        x += 205

    image.save(plot_path)


def draw_aggregate_plot(results, plot_path):
    ranked = sorted(results, key=lambda item: (seeker_win_rate(item), metric(item, "mean_visible_fraction")), reverse=True)
    image = Image.new("RGB", (1240, 820), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    title_font = load_font(22, bold=True)
    label_font = load_font(13)
    small_font = load_font(11)

    draw.text((30, 24), "Policy Benchmark Summary", fill=(20, 20, 20), font=title_font)
    draw.text(
        (30, 54),
        "Bars show seeker win rate, hider win rate, tie rate, and mean visible fraction.",
        fill=(55, 55, 55),
        font=label_font,
    )

    chart = (285, 120, 1160, 735)
    x0, y0, x1, y1 = chart
    draw.rectangle(chart, outline=(45, 45, 45), width=1)
    for idx in range(6):
        frac = idx / 5.0
        x = x0 + frac * (x1 - x0)
        draw.line((x, y0, x, y1), fill=(228, 228, 228))
        draw.text((x - 10, y1 + 8), "{:.1f}".format(frac), fill=(75, 75, 75), font=small_font)

    row_h = (y1 - y0) / max(len(ranked), 1)
    for idx, result in enumerate(ranked):
        row_y = y0 + idx * row_h
        label = display_name(result["policy_rel"])
        draw.text((30, row_y + 12), label[:31], fill=(30, 30, 30), font=label_font)
        draw.text((30, row_y + 33), short_env(result["env"]), fill=(90, 90, 90), font=small_font)

        win = seeker_win_rate(result)
        hider = hider_win_rate(result)
        ties = tie_rate(result)
        vis = metric(result, "mean_visible_fraction")
        win_w = win * (x1 - x0)
        hider_w = hider * (x1 - x0)
        tie_w = ties * (x1 - x0)
        vis_w = vis * (x1 - x0)
        draw.rectangle((x0, row_y + 7, x0 + win_w, row_y + 18), fill=(195, 76, 65))
        draw.rectangle((x0, row_y + 22, x0 + hider_w, row_y + 33), fill=(53, 150, 83))
        draw.rectangle((x0, row_y + 37, x0 + tie_w, row_y + 48), fill=(135, 135, 135))
        draw.rectangle((x0, row_y + 52, x0 + vis_w, row_y + 63), fill=(56, 98, 170))
        draw.text((x0 + win_w + 5, row_y + 3), "{:.2f}".format(win), fill=(195, 76, 65), font=small_font)
        draw.text((x0 + hider_w + 5, row_y + 18), "{:.2f}".format(hider), fill=(53, 150, 83), font=small_font)
        draw.text((x0 + tie_w + 5, row_y + 33), "{:.2f}".format(ties), fill=(100, 100, 100), font=small_font)
        draw.text((x0 + vis_w + 5, row_y + 48), "{:.2f}".format(vis), fill=(56, 98, 170), font=small_font)

    legend_x = 700
    legend_y = 34
    legend = [
        ((195, 76, 65), "seeker win rate"),
        ((53, 150, 83), "hider win rate"),
        ((135, 135, 135), "tie rate"),
        ((56, 98, 170), "visible fraction"),
    ]
    for idx, (color, label) in enumerate(legend):
        x = legend_x + (idx % 2) * 190
        y = legend_y + (idx // 2) * 24
        draw.rectangle((x, y, x + 15, y + 15), fill=color)
        draw.text((x + 22, y - 3), label, fill=(35, 35, 35), font=label_font)

    image.save(plot_path)


def write_text_report(results, args, report_path):
    ranked = sorted(results, key=lambda item: (seeker_win_rate(item), metric(item, "mean_visible_fraction")), reverse=True)
    lines = []
    lines.append("Policy Benchmark Report")
    lines.append("created_unix: {:.3f}".format(time.time()))
    lines.append("env_mode: {}".format(args.env))
    lines.append("episodes_per_policy: {}".format(args.episodes))
    lines.append("steps: {}".format(args.steps))
    lines.append("seed_start: {}".format(args.seed))
    lines.append("")
    lines.append(
        "{:<4} {:<42} {:<28} {:>4} {:>4} {:>4} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
            "rank", "policy", "env", "S", "H", "T", "S_win", "H_win", "T_rate",
            "visible", "S_return", "H_return", "plot"
        )
    )
    lines.append("-" * 170)
    for idx, result in enumerate(ranked, start=1):
        lines.append(
            "{:<4} {:<42} {:<28} {:>4} {:>4} {:>4} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.3f} {:>10.3f} {:>10}".format(
                idx,
                display_name(result["policy_rel"])[:42],
                short_env(result["env"])[:28],
                int(result.get("seeker_wins", 0)),
                int(result.get("hider_wins", 0)),
                int(result.get("ties", 0)),
                seeker_win_rate(result),
                hider_win_rate(result),
                tie_rate(result),
                metric(result, "mean_visible_fraction"),
                metric(result, "mean_seeker_return"),
                metric(result, "mean_hider_return"),
                os.path.basename(result["plot_path"]),
            )
        )
    lines.append("")
    lines.append("Plot directory: {}".format(args.plot_dir))
    lines.append("JSON summary: {}".format(args.summary_out))

    with open(report_path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="auto")
    parser.add_argument("--quadrant-env", default=QUADRANT_ENV)
    parser.add_argument("--full-env", default=FULL_ENV)
    parser.add_argument("--policy-root", default="/workspace/multi-agent-emergence-environments/examples")
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out-dir", default="/workspace/runs/policy_benchmark")
    parser.add_argument("--plot-dir", default="/workspace/plots/policy_benchmark")
    parser.add_argument("--report-out", default="/workspace/runs/policy_benchmark/benchmark_report.txt")
    parser.add_argument("--summary-out", default="/workspace/runs/policy_benchmark/benchmark_summary.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    if os.path.dirname(args.report_out):
        os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    if os.path.dirname(args.summary_out):
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)

    policies = args.policy or DEFAULT_POLICIES
    results = []
    failures = []

    for policy_rel in policies:
        name = sanitize_name(policy_rel)
        policy_path = policy_rel if os.path.isabs(policy_rel) else os.path.join(args.policy_root, policy_rel)
        env_path = select_env(policy_rel, args.env, args.quadrant_env, args.full_env)
        eval_path = os.path.join(args.out_dir, "{}.json".format(name))
        plot_path = os.path.join(args.plot_dir, "{}.png".format(name))

        print("=== benchmark: {} ===".format(policy_rel))
        print("env:", env_path)
        command = [
            sys.executable,
            "/workspace/scripts/evaluate_policy.py",
            "--env", env_path,
            "--policy", policy_path,
            "--episodes", str(args.episodes),
            "--steps", str(args.steps),
            "--seed", str(args.seed),
            "--out", eval_path,
        ]
        code, output = run_command(command)
        print(tail(output))
        if code != 0 or not os.path.exists(eval_path):
            failures.append({
                "policy_rel": policy_rel,
                "policy_path": policy_path,
                "env": env_path,
                "returncode": code,
                "error_tail": tail(output),
            })
            print("status: failed")
            continue

        result = load_eval_json(eval_path)
        result["policy_rel"] = policy_rel
        result["policy_path"] = policy_path
        result["env"] = env_path
        result["eval_path"] = eval_path
        result["plot_path"] = plot_path
        draw_policy_plot(result, plot_path)
        print("plot:", plot_path)
        results.append(result)

    aggregate_plot = os.path.join(args.plot_dir, "aggregate_summary.png")
    if results:
        draw_aggregate_plot(results, aggregate_plot)
        write_text_report(results, args, args.report_out)

    summary = {
        "env": args.env,
        "quadrant_env": args.quadrant_env,
        "full_env": args.full_env,
        "episodes_per_policy": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "results": results,
        "failures": failures,
        "report_out": args.report_out,
        "summary_out": args.summary_out,
        "plot_dir": args.plot_dir,
        "aggregate_plot": aggregate_plot,
    }
    with open(args.summary_out, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print("report:", args.report_out)
    print("summary:", args.summary_out)
    print("plot_dir:", args.plot_dir)
    print("aggregate_plot:", aggregate_plot)
    print("policy_count:", len(results))
    print("failed_count:", len(failures))


if __name__ == "__main__":
    main()
