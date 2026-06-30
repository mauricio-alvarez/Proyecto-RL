import argparse
import json
import os
import re
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont


FULL_ENV = "/workspace/multi-agent-emergence-environments/examples/hide_and_seek_full.jsonnet"
QUADRANT_ENV = "/workspace/multi-agent-emergence-environments/examples/hide_and_seek_quadrant.jsonnet"

FULL_POLICIES = [
    "hide_and_seek_full.npz",
    "hide_and_seek_policy_phases/a_chasing.npz",
    "hide_and_seek_policy_phases/b_forts.npz",
    "hide_and_seek_policy_phases/c_ramps.npz",
    "hide_and_seek_policy_phases/d_ramp_defense.npz",
    "hide_and_seek_policy_phases/e_box_surfing.npz",
]

QUADRANT_POLICIES = [
    "hide_and_seek_quadrant.npz",
    "hide_and_seek_quadrant_physics_exploits.npz",
]


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


def policy_path(policy_root, policy_rel):
    return policy_rel if os.path.isabs(policy_rel) else os.path.join(policy_root, policy_rel)


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


def load_json(path):
    with open(path) as input_file:
        return json.load(input_file)


def rate(result, key):
    episodes = max(int(result.get("episodes", 0)), 1)
    return float(result.get(key, 0)) / episodes


def seeker_win_rate(result):
    return rate(result, "seeker_wins")


def hider_win_rate(result):
    return rate(result, "hider_wins")


def tie_rate(result):
    return rate(result, "ties")


def metric(result, key, default=0.0):
    value = result.get(key, default)
    return default if value is None else value


def color_scale(value, low_color, high_color):
    value = max(0.0, min(1.0, value))
    return tuple(
        int(low_color[idx] + value * (high_color[idx] - low_color[idx]))
        for idx in range(3)
    )


def draw_rotated_text(base, text, position, font, fill=(30, 30, 30)):
    temp = Image.new("RGBA", (220, 34), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp)
    draw.text((0, 7), text[:24], font=font, fill=fill)
    rotated = temp.rotate(45, expand=True)
    base.paste(rotated, position, rotated)


def label_lines(label, max_line_length=15, max_lines=3):
    parts = label.split("_")
    lines = []
    current = ""
    for part in parts:
        candidate = part if not current else "{}_{}".format(current, part)
        if len(candidate) <= max_line_length:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)

    if len(lines) <= max_lines:
        return lines

    head = lines[:max_lines - 1]
    tail = "_".join(lines[max_lines - 1:])
    if len(tail) > max_line_length:
        tail = tail[:max_line_length - 1] + "."
    return head + [tail]


def draw_centered_lines(draw, lines, center_x, top_y, font, fill):
    line_height = 15
    for idx, line in enumerate(lines):
        text_w, _ = draw.textsize(line, font=font)
        draw.text((center_x - text_w / 2, top_y + idx * line_height), line, fill=fill, font=font)


def draw_heatmap(matrix, hider_policies, seeker_policies, key, title, path, reverse=False):
    cell_w = 135
    cell_h = 74
    left = 260
    top = 190
    width = left + cell_w * len(seeker_policies) + 60
    height = top + cell_h * len(hider_policies) + 80
    image = Image.new("RGB", (max(width, 920), max(height, 620)), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    title_font = load_font(22, bold=True)
    label_font = load_font(13)
    small_font = load_font(11)

    draw.text((30, 24), title, fill=(20, 20, 20), font=title_font)
    draw.text((30, 58), "rows = hider policy, columns = seeker policy", fill=(65, 65, 65), font=label_font)
    draw.text((left, 120), "seeker policy", fill=(45, 45, 45), font=label_font)
    draw.text((30, top - 28), "hider policy", fill=(45, 45, 45), font=label_font)

    for col, policy in enumerate(seeker_policies):
        center_x = left + col * cell_w + (cell_w - 6) / 2
        draw_centered_lines(
            draw,
            label_lines(display_name(policy), max_line_length=14, max_lines=3),
            center_x,
            140,
            small_font,
            (30, 30, 30),
        )

    for row, policy in enumerate(hider_policies):
        y = top + row * cell_h + 25
        draw.text((30, y), display_name(policy)[:28], fill=(30, 30, 30), font=label_font)

    low = (245, 245, 245)
    high = (195, 76, 65) if not reverse else (53, 150, 83)
    for row, hider_policy in enumerate(hider_policies):
        for col, seeker_policy in enumerate(seeker_policies):
            item = matrix.get((hider_policy, seeker_policy))
            value = 0.0 if item is None else float(item.get(key, 0.0))
            x0 = left + col * cell_w
            y0 = top + row * cell_h
            x1 = x0 + cell_w - 6
            y1 = y0 + cell_h - 6
            draw.rectangle((x0, y0, x1, y1), fill=color_scale(value, low, high), outline=(80, 80, 80))
            draw.text((x0 + 27, y0 + 20), "{:.2f}".format(value), fill=(20, 20, 20), font=label_font)

    image.save(path)


def draw_crossplay_summary(results, hider_policies, seeker_policies, plot_dir):
    matrix = {
        (result["hider_policy_rel"], result["seeker_policy_rel"]): result
        for result in results
    }
    seeker_path = os.path.join(plot_dir, "crossplay_seeker_win_rate.png")
    hider_path = os.path.join(plot_dir, "crossplay_hider_win_rate.png")
    visible_path = os.path.join(plot_dir, "crossplay_visible_fraction.png")
    draw_heatmap(matrix, hider_policies, seeker_policies, "seeker_win_rate", "Cross-Play Seeker Win Rate", seeker_path)
    draw_heatmap(matrix, hider_policies, seeker_policies, "hider_win_rate", "Cross-Play Hider Win Rate", hider_path, reverse=True)
    draw_heatmap(matrix, hider_policies, seeker_policies, "mean_visible_fraction", "Cross-Play Visible Fraction", visible_path)
    return {
        "seeker_win_rate": seeker_path,
        "hider_win_rate": hider_path,
        "visible_fraction": visible_path,
    }


def write_report(results, args, report_path):
    ranked_seekers = sorted(results, key=lambda item: (item["seeker_win_rate"], item["mean_visible_fraction"]), reverse=True)
    ranked_hiders = sorted(results, key=lambda item: (item["hider_win_rate"], -item["mean_visible_fraction"]), reverse=True)

    lines = []
    lines.append("Cross-Play Benchmark Report")
    lines.append("created_unix: {:.3f}".format(time.time()))
    lines.append("env: {}".format(args.env))
    lines.append("episodes_per_matchup: {}".format(args.episodes))
    lines.append("steps: {}".format(args.steps))
    lines.append("seed_start: {}".format(args.seed))
    lines.append("")
    lines.append("Top seeker matchups")
    lines.append(
        "{:<4} {:<32} {:<32} {:>4} {:>4} {:>4} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
            "rank", "hider_policy", "seeker_policy", "S", "H", "T", "S_win", "H_win",
            "T_rate", "visible", "S_return", "H_return"
        )
    )
    lines.append("-" * 155)
    for idx, result in enumerate(ranked_seekers, start=1):
        lines.append(format_report_row(idx, result))

    lines.append("")
    lines.append("Top hider matchups")
    lines.append(
        "{:<4} {:<32} {:<32} {:>4} {:>4} {:>4} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
            "rank", "hider_policy", "seeker_policy", "S", "H", "T", "S_win", "H_win",
            "T_rate", "visible", "S_return", "H_return"
        )
    )
    lines.append("-" * 155)
    for idx, result in enumerate(ranked_hiders, start=1):
        lines.append(format_report_row(idx, result))

    lines.append("")
    lines.append("Plot directory: {}".format(args.plot_dir))
    lines.append("JSON summary: {}".format(args.summary_out))

    with open(report_path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def format_report_row(idx, result):
    return "{:<4} {:<32} {:<32} {:>4} {:>4} {:>4} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.3f} {:>10.3f}".format(
        idx,
        display_name(result["hider_policy_rel"])[:32],
        display_name(result["seeker_policy_rel"])[:32],
        int(result.get("seeker_wins", 0)),
        int(result.get("hider_wins", 0)),
        int(result.get("ties", 0)),
        result["seeker_win_rate"],
        result["hider_win_rate"],
        result["tie_rate"],
        result["mean_visible_fraction"],
        result["mean_seeker_return"],
        result["mean_hider_return"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["full", "quadrant"], default="full")
    parser.add_argument("--full-env", default=FULL_ENV)
    parser.add_argument("--quadrant-env", default=QUADRANT_ENV)
    parser.add_argument("--policy-root", default="/workspace/multi-agent-emergence-environments/examples")
    parser.add_argument("--hider-policy", action="append", default=[])
    parser.add_argument("--seeker-policy", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out-dir", default="/workspace/runs/crossplay_benchmark")
    parser.add_argument("--plot-dir", default="/workspace/plots/crossplay_benchmark")
    parser.add_argument("--report-out", default="/workspace/runs/crossplay_benchmark/crossplay_report.txt")
    parser.add_argument("--summary-out", default="/workspace/runs/crossplay_benchmark/crossplay_summary.json")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    if os.path.dirname(args.report_out):
        os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    if os.path.dirname(args.summary_out):
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)

    env_path = args.full_env if args.env == "full" else args.quadrant_env
    default_policies = FULL_POLICIES if args.env == "full" else QUADRANT_POLICIES
    hider_policies = args.hider_policy or default_policies
    seeker_policies = args.seeker_policy or default_policies

    results = []
    failures = []

    for hider_policy_rel in hider_policies:
        for seeker_policy_rel in seeker_policies:
            matchup_name = "{}__vs__{}".format(
                sanitize_name(hider_policy_rel),
                sanitize_name(seeker_policy_rel),
            )
            out_path = os.path.join(args.out_dir, "{}.json".format(matchup_name))
            print("=== crossplay: hider={} seeker={} ===".format(hider_policy_rel, seeker_policy_rel))

            if args.reuse_existing and os.path.exists(out_path):
                print("status: reusing {}".format(out_path))
            else:
                command = [
                    sys.executable,
                    "/workspace/scripts/evaluate_crossplay.py",
                    "--env", env_path,
                    "--hider-policy", policy_path(args.policy_root, hider_policy_rel),
                    "--seeker-policy", policy_path(args.policy_root, seeker_policy_rel),
                    "--episodes", str(args.episodes),
                    "--steps", str(args.steps),
                    "--seed", str(args.seed),
                    "--out", out_path,
                ]
                code, output = run_command(command)
                print(tail(output))
                if code != 0 or not os.path.exists(out_path):
                    failures.append({
                        "hider_policy_rel": hider_policy_rel,
                        "seeker_policy_rel": seeker_policy_rel,
                        "returncode": code,
                        "error_tail": tail(output),
                    })
                    print("status: failed")
                    continue

            result = load_json(out_path)
            result["hider_policy_rel"] = hider_policy_rel
            result["seeker_policy_rel"] = seeker_policy_rel
            result["env"] = env_path
            result["eval_path"] = out_path
            result["seeker_win_rate"] = seeker_win_rate(result)
            result["hider_win_rate"] = hider_win_rate(result)
            result["tie_rate"] = tie_rate(result)
            results.append(result)
            print(
                "status: done seeker_win_rate={:.3f} hider_win_rate={:.3f} tie_rate={:.3f}".format(
                    result["seeker_win_rate"],
                    result["hider_win_rate"],
                    result["tie_rate"],
                )
            )

    plot_paths = {}
    if results:
        plot_paths = draw_crossplay_summary(results, hider_policies, seeker_policies, args.plot_dir)
        write_report(results, args, args.report_out)

    summary = {
        "env": args.env,
        "env_path": env_path,
        "episodes_per_matchup": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "hider_policies": hider_policies,
        "seeker_policies": seeker_policies,
        "results": results,
        "failures": failures,
        "plot_paths": plot_paths,
        "report_out": args.report_out,
        "summary_out": args.summary_out,
    }
    with open(args.summary_out, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print("report:", args.report_out)
    print("summary:", args.summary_out)
    print("plot_dir:", args.plot_dir)
    print("matchup_count:", len(results))
    print("failed_count:", len(failures))


if __name__ == "__main__":
    main()
