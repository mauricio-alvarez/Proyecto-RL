import argparse
import csv
import html
import json
import os


def read_progress(path):
    rows = []
    with open(path, "r") as input_file:
        for row in csv.DictReader(input_file):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    if not rows:
        raise ValueError("No rows found in {}".format(path))
    return rows


def series(rows, key):
    return [float(row[key]) for row in rows]


def plot_png(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = series(rows, "update")
    plots = []

    def save_plot(filename, title, y_items, ylabel):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
        for key, label in y_items:
            ax.plot(updates, series(rows, key), label=label, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("PPO update")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        plots.append(path)

    save_plot(
        "ppo_returns.png",
        "Recent Mean Team Returns",
        [
            ("recent_mean_hider_return", "hiders"),
            ("recent_mean_seeker_return", "seekers"),
        ],
        "mean return",
    )
    save_plot(
        "ppo_recent_wins.png",
        "Recent Wins In Last 20 Episodes",
        [
            ("recent_hider_wins", "hider wins"),
            ("recent_seeker_wins", "seeker wins"),
            ("recent_ties", "ties"),
        ],
        "count",
    )
    save_plot(
        "ppo_optimization.png",
        "PPO Optimization Diagnostics",
        [
            ("approx_kl", "approx KL"),
            ("clip_fraction", "clip fraction"),
        ],
        "value",
    )
    save_plot(
        "ppo_entropy.png",
        "Policy Entropy",
        [
            ("entropy", "entropy"),
        ],
        "entropy",
    )
    save_plot(
        "ppo_value_loss.png",
        "Value Loss",
        [
            ("value_loss", "value loss"),
        ],
        "loss",
    )
    return plots


def plot_svg(rows, out_dir):
    updates = series(rows, "update")
    plots = []
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    def points_for(xs, ys, width, height, margin, y_min, y_max):
        x_min = min(xs)
        x_max = max(xs)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0
        points = []
        for x_value, y_value in zip(xs, ys):
            x = margin + (x_value - x_min) / (x_max - x_min) * (width - 2 * margin)
            y = height - margin - (y_value - y_min) / (y_max - y_min) * (height - 2 * margin)
            points.append("{:.2f},{:.2f}".format(x, y))
        return " ".join(points)

    def save_plot(filename, title, y_items, ylabel):
        width = 1000
        height = 520
        margin = 70
        y_values = []
        for key, _ in y_items:
            y_values.extend(series(rows, key))
        y_min = min(y_values)
        y_max = max(y_values)
        padding = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= padding
        y_max += padding

        body = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(w=width, h=height),
            '<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>'.format(w=width, h=height),
            '<text x="{x}" y="32" text-anchor="middle" font-family="Arial" font-size="22" fill="#111827">{}</text>'.format(html.escape(title), x=width / 2),
            '<line x1="{m}" y1="{y}" x2="{x2}" y2="{y}" stroke="#111827" stroke-width="1.5"/>'.format(m=margin, y=height - margin, x2=width - margin),
            '<line x1="{m}" y1="{m}" x2="{m}" y2="{y2}" stroke="#111827" stroke-width="1.5"/>'.format(m=margin, y2=height - margin),
            '<text x="{x}" y="{y}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">PPO update</text>'.format(x=width / 2, y=height - 20),
            '<text x="18" y="{y}" transform="rotate(-90 18,{y})" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">{}</text>'.format(html.escape(ylabel), y=height / 2),
            '<text x="{x}" y="{y}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{:.3f}</text>'.format(y_max, x=margin - 8, y=margin + 4),
            '<text x="{x}" y="{y}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{:.3f}</text>'.format(y_min, x=margin - 8, y=height - margin + 4),
        ]

        for idx, (key, label) in enumerate(y_items):
            values = series(rows, key)
            color = colors[idx % len(colors)]
            points = points_for(updates, values, width, height, margin, y_min, y_max)
            legend_y = 62 + idx * 22
            body.append('<polyline fill="none" stroke="{}" stroke-width="2.2" points="{}"/>'.format(color, points))
            body.append('<line x1="{x}" y1="{y}" x2="{x2}" y2="{y}" stroke="{c}" stroke-width="3"/>'.format(x=width - 220, x2=width - 190, y=legend_y, c=color))
            body.append('<text x="{x}" y="{y}" font-family="Arial" font-size="14" fill="#111827">{}</text>'.format(html.escape(label), x=width - 180, y=legend_y + 4))
        body.append("</svg>")

        path = os.path.join(out_dir, filename)
        with open(path, "w") as output_file:
            output_file.write("\n".join(body))
            output_file.write("\n")
        plots.append(path)

    save_plot(
        "ppo_returns.svg",
        "Recent Mean Team Returns",
        [
            ("recent_mean_hider_return", "hiders"),
            ("recent_mean_seeker_return", "seekers"),
        ],
        "mean return",
    )
    save_plot(
        "ppo_recent_wins.svg",
        "Recent Wins In Last 20 Episodes",
        [
            ("recent_hider_wins", "hider wins"),
            ("recent_seeker_wins", "seeker wins"),
            ("recent_ties", "ties"),
        ],
        "count",
    )
    save_plot(
        "ppo_optimization.svg",
        "PPO Optimization Diagnostics",
        [
            ("approx_kl", "approx KL"),
            ("clip_fraction", "clip fraction"),
        ],
        "value",
    )
    save_plot(
        "ppo_entropy.svg",
        "Policy Entropy",
        [
            ("entropy", "entropy"),
        ],
        "entropy",
    )
    save_plot(
        "ppo_value_loss.svg",
        "Value Loss",
        [
            ("value_loss", "value loss"),
        ],
        "loss",
    )
    return plots


def write_report(path, rows, plots):
    first = rows[0]
    last = rows[-1]
    summary = {
        "updates": int(last["update"]),
        "episodes_completed": int(last["episodes_completed"]),
        "first": first,
        "last": last,
        "plots": plots,
    }
    with open(path, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_progress(args.progress)
    try:
        plots = plot_png(rows, args.out_dir)
    except ImportError:
        plots = plot_svg(rows, args.out_dir)
    report_path = args.report_out or os.path.join(args.out_dir, "plot_summary.json")
    write_report(report_path, rows, plots)

    print("saved_report:", report_path, flush=True)
    for path in plots:
        print("saved_plot:", path, flush=True)


if __name__ == "__main__":
    main()
