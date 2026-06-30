import argparse
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


AGENT_RADIUS = 8
BOX_HALF = 12
RAMP_HALF_W = 18
RAMP_HALF_H = 6


def normalize_layout(layout):
    if layout is None:
        return None
    if isinstance(layout, np.ndarray) and layout.shape == ():
        layout = layout.item()
    if not isinstance(layout, dict):
        return None
    return layout


def get_episode_layout(episode):
    layout = normalize_layout(episode.get("layout"))
    if layout is not None:
        return layout
    return None


def load_default_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except IOError:
        return ImageFont.load_default()


def iter_positions(episode):
    for obs in episode["observations"]:
        yield obs["observation_self"][:, :2]
        if "box_obs" in obs:
            yield obs["box_obs"][0, :, :2]
        if "ramp_obs" in obs:
            yield obs["ramp_obs"][0, :, :2]


def compute_bounds(episode, layout=None, margin=0.4):
    points = [positions for positions in iter_positions(episode) if positions.size]
    all_points = np.concatenate(points, axis=0)
    low = np.floor(all_points.min(axis=0) - margin)
    high = np.ceil(all_points.max(axis=0) + margin)

    floor_size = None
    layout = normalize_layout(layout)
    if layout is not None:
        floor_size = layout.get("floor_size")
    if floor_size is None:
        floor_size = 6.0

    low = np.minimum(low, np.array([0.0, 0.0]))
    high = np.maximum(high, np.array([float(floor_size), float(floor_size)]))
    return low, high


def world_to_pixel(point, low, high, width, height, pad):
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    scale = min(usable_w / (high[0] - low[0]), usable_h / (high[1] - low[1]))
    x = pad + (point[0] - low[0]) * scale
    y = height - pad - (point[1] - low[1]) * scale
    return int(round(x)), int(round(y))


def draw_grid(draw, low, high, width, height, pad, color=(225, 225, 225)):
    for value in range(int(low[0]), int(high[0]) + 1):
        x0, y0 = world_to_pixel((value, low[1]), low, high, width, height, pad)
        x1, y1 = world_to_pixel((value, high[1]), low, high, width, height, pad)
        draw.line((x0, y0, x1, y1), fill=color)
    for value in range(int(low[1]), int(high[1]) + 1):
        x0, y0 = world_to_pixel((low[0], value), low, high, width, height, pad)
        x1, y1 = world_to_pixel((high[0], value), low, high, width, height, pad)
        draw.line((x0, y0, x1, y1), fill=color)


def draw_walls(draw, layout, low, high, width, height, pad):
    layout = normalize_layout(layout)
    if layout is None:
        return False

    wall_geoms = layout.get("wall_geoms", [])
    if not wall_geoms:
        return False

    for wall in wall_geoms:
        pos = np.asarray(wall["pos"], dtype=np.float32)
        size = np.asarray(wall["size"], dtype=np.float32)
        wall_low = pos[:2] - size[:2]
        wall_high = pos[:2] + size[:2]
        x0, y0 = world_to_pixel(wall_low, low, high, width, height, pad)
        x1, y1 = world_to_pixel(wall_high, low, high, width, height, pad)
        draw.rectangle(
            (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
            fill=(98, 145, 92),
            outline=(56, 100, 58),
        )

    return True


def draw_entity_label(draw, xy, label, font, fill=(20, 20, 20)):
    x, y = xy
    draw.text((x + 9, y - 17), label, fill=fill, font=font)


def draw_frame(episode, episode_idx, step_idx, low, high, width, height, pad, n_hiders, tail,
               layout):
    obs = episode["observations"][step_idx]
    rewards = np.asarray(episode["rewards"][step_idx])
    done = episode["dones"][step_idx]
    info = episode["infos"][step_idx]

    image = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    font = load_default_font()

    draw_grid(draw, low, high, width, height, pad)
    walls_drawn = draw_walls(draw, layout, low, high, width, height, pad)

    x0, y0 = world_to_pixel(low, low, high, width, height, pad)
    x1, y1 = world_to_pixel(high, low, high, width, height, pad)
    draw.rectangle((x0, y1, x1, y0), outline=(30, 30, 30), width=2)

    if "box_obs" in obs:
        for box_idx, point in enumerate(obs["box_obs"][0, :, :2]):
            x, y = world_to_pixel(point, low, high, width, height, pad)
            draw.rectangle(
                (x - BOX_HALF, y - BOX_HALF, x + BOX_HALF, y + BOX_HALF),
                fill=(190, 150, 95),
                outline=(95, 70, 35),
                width=2,
            )
            draw.text((x - 4, y - 8), "B", fill=(40, 25, 10), font=font)

    if "ramp_obs" in obs:
        for ramp_idx, point in enumerate(obs["ramp_obs"][0, :, :2]):
            x, y = world_to_pixel(point, low, high, width, height, pad)
            draw.rectangle(
                (x - RAMP_HALF_W, y - RAMP_HALF_H, x + RAMP_HALF_W, y + RAMP_HALF_H),
                fill=(90, 170, 100),
                outline=(30, 90, 40),
                width=2,
            )
            draw.text((x - 5, y - 9), "R", fill=(10, 55, 20), font=font)

    agent_positions = obs["observation_self"][:, :2]
    first_tail_step = max(0, step_idx - tail)
    for agent_idx in range(agent_positions.shape[0]):
        trail = []
        for prev_step in range(first_tail_step, step_idx + 1):
            prev_pos = episode["observations"][prev_step]["observation_self"][agent_idx, :2]
            trail.append(world_to_pixel(prev_pos, low, high, width, height, pad))
        if len(trail) > 1:
            draw.line(trail, fill=(130, 130, 130), width=2)

    for agent_idx, point in enumerate(agent_positions):
        x, y = world_to_pixel(point, low, high, width, height, pad)
        is_hider = agent_idx < n_hiders
        fill = (70, 165, 95) if is_hider else (205, 85, 70)
        outline = (20, 85, 35) if is_hider else (110, 35, 30)
        label = "H{}".format(agent_idx) if is_hider else "S{}".format(agent_idx - n_hiders)
        draw.ellipse(
            (x - AGENT_RADIUS, y - AGENT_RADIUS, x + AGENT_RADIUS, y + AGENT_RADIUS),
            fill=fill,
            outline=outline,
            width=2,
        )
        draw_entity_label(draw, (x, y), label, font)

    summary = [
        "episode={} step={}/{}".format(episode_idx, step_idx + 1, len(episode["observations"])),
        "reward={}".format(np.round(rewards, 2).tolist()),
        "done={} discard={}".format(done, info.get("discard_episode")),
    ]
    if not walls_drawn:
        summary.append("walls=not stored in rollout; regenerate rollout with current collector")
    for line_idx, line in enumerate(summary):
        draw.text((14, 12 + line_idx * 18), line, fill=(20, 20, 20), font=font)

    legend_y = height - 58
    draw.ellipse((14, legend_y, 28, legend_y + 14), fill=(70, 165, 95), outline=(20, 85, 35))
    draw.text((34, legend_y - 2), "Hider", fill=(20, 20, 20), font=font)
    draw.ellipse((100, legend_y, 114, legend_y + 14), fill=(205, 85, 70), outline=(110, 35, 30))
    draw.text((120, legend_y - 2), "Seeker", fill=(20, 20, 20), font=font)
    draw.rectangle((200, legend_y, 214, legend_y + 14), fill=(190, 150, 95), outline=(95, 70, 35))
    draw.text((220, legend_y - 2), "Box", fill=(20, 20, 20), font=font)
    draw.rectangle((270, legend_y + 4, 294, legend_y + 10), fill=(90, 170, 100), outline=(30, 90, 40))
    draw.text((300, legend_y - 2), "Ramp", fill=(20, 20, 20), font=font)
    draw.rectangle((370, legend_y, 384, legend_y + 14), fill=(98, 145, 92), outline=(56, 100, 58))
    draw.text((390, legend_y - 2), "Wall", fill=(20, 20, 20), font=font)

    return np.asarray(image)


def write_video(frames, out_path, fps):
    ext = os.path.splitext(out_path)[1].lower()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if ext == ".gif":
        imageio.mimsave(out_path, frames, duration=1.0 / fps)
        return

    try:
        imageio.mimsave(out_path, frames, fps=fps)
    except Exception as exc:
        fallback = os.path.splitext(out_path)[0] + ".gif"
        imageio.mimsave(fallback, frames, duration=1.0 / fps)
        print("Could not write {}: {}".format(out_path, exc))
        print("Wrote GIF fallback:", fallback)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/workspace/runs/random_rollouts_hide_seek_quadrant.npz")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", default="/workspace/videos/random_rollout_ep0.gif")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--pad", type=int, default=60)
    parser.add_argument("--n-hiders", type=int, default=2)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    episodes = data["episodes"]
    if args.episode < 0 or args.episode >= len(episodes):
        raise ValueError("episode index {} outside [0, {})".format(args.episode, len(episodes)))

    episode = episodes[args.episode]
    layout = get_episode_layout(episode)
    low, high = compute_bounds(episode, layout=layout)
    n_frames = len(episode["observations"])
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    frames = [
        draw_frame(
            episode=episode,
            episode_idx=args.episode,
            step_idx=step_idx,
            low=low,
            high=high,
            width=args.width,
            height=args.height,
            pad=args.pad,
            n_hiders=args.n_hiders,
            tail=args.tail,
            layout=layout,
        )
        for step_idx in range(n_frames)
    ]

    write_video(frames, args.out, args.fps)
    print("input:", args.input)
    print("episode:", args.episode)
    print("frames:", len(frames))
    print("fps:", args.fps)
    print("output:", args.out)
    print("walls_drawn:", layout is not None and bool(layout.get("wall_geoms")))


if __name__ == "__main__":
    main()
