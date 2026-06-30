import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf


MOVEMENT_CLASSES = 11
BINARY_CLASSES = 2


def parse_hidden_sizes(value):
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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


def select_obs_keys(data, requested_keys):
    available = sorted(key for key in data.files if key.startswith("obs_"))
    if not requested_keys:
        return available

    selected = []
    for key in requested_keys:
        normalized = key if key.startswith("obs_") else "obs_{}".format(key)
        if normalized not in data.files:
            raise ValueError("Observation key not found: {}".format(normalized))
        selected.append(normalized)
    return selected


def flatten_feature(array):
    array = np.asarray(array)
    return array.reshape((array.shape[0], -1)).astype(np.float32)


def build_features(data, obs_keys, include_role):
    features = [flatten_feature(data[key]) for key in obs_keys]
    if include_role:
        role = data["role"].astype(np.int32)
        role_onehot = np.eye(2, dtype=np.float32)[role]
        features.append(role_onehot)
    return np.concatenate(features, axis=1)


def role_mask(data, role_name):
    if role_name == "all":
        return np.ones(len(data["role"]), dtype=bool)
    role_id = 0 if role_name == "hider" else 1
    return data["role"] == role_id


def split_indices_by_episode(data, mask, train_frac, seed):
    rng = np.random.RandomState(seed)
    candidate_indices = np.where(mask)[0]
    episodes = np.unique(data["episode"][candidate_indices])
    rng.shuffle(episodes)

    if len(episodes) >= 2:
        n_train = int(round(len(episodes) * train_frac))
        n_train = max(1, min(n_train, len(episodes) - 1))
        train_episodes = set(episodes[:n_train].tolist())
        train_mask = np.array([episode in train_episodes for episode in data["episode"]])
        train_idx = np.where(mask & train_mask)[0]
        val_idx = np.where(mask & ~train_mask)[0]
        return train_idx, val_idx, {
            "split_type": "episode",
            "train_episodes": sorted([int(value) for value in train_episodes]),
            "val_episodes": sorted([int(value) for value in episodes[n_train:].tolist()]),
        }

    shuffled = candidate_indices.copy()
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_frac))
    n_train = max(1, min(n_train, len(shuffled) - 1))
    return shuffled[:n_train], shuffled[n_train:], {
        "split_type": "sample",
        "train_episodes": sorted([int(value) for value in episodes.tolist()]),
        "val_episodes": sorted([int(value) for value in episodes.tolist()]),
    }


def normalize_features(x, train_idx):
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def labels_from_data(data):
    movement = data["action_action_movement"].astype(np.int32)
    pull = data["action_action_pull"].astype(np.int32).reshape(-1)
    glueall = data["action_action_glueall"].astype(np.int32).reshape(-1)
    return {
        "movement_0": movement[:, 0],
        "movement_1": movement[:, 1],
        "movement_2": movement[:, 2],
        "pull": pull,
        "glueall": glueall,
    }


def batch_iter(indices, batch_size, rng, shuffle=True):
    indices = indices.copy()
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def build_model(input_dim, hidden_sizes, learning_rate, dropout):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, input_dim], name="features")
    is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
    labels = {
        "movement_0": tf.placeholder(tf.int32, shape=[None], name="label_movement_0"),
        "movement_1": tf.placeholder(tf.int32, shape=[None], name="label_movement_1"),
        "movement_2": tf.placeholder(tf.int32, shape=[None], name="label_movement_2"),
        "pull": tf.placeholder(tf.int32, shape=[None], name="label_pull"),
        "glueall": tf.placeholder(tf.int32, shape=[None], name="label_glueall"),
    }

    h = x
    for idx, size in enumerate(hidden_sizes):
        h = tf.layers.dense(h, size, activation=tf.nn.relu, name="hidden_{}".format(idx))
        if dropout > 0.0:
            h = tf.layers.dropout(h, rate=dropout, training=is_training, name="dropout_{}".format(idx))

    logits = {
        "movement_0": tf.layers.dense(h, MOVEMENT_CLASSES, name="logits_movement_0"),
        "movement_1": tf.layers.dense(h, MOVEMENT_CLASSES, name="logits_movement_1"),
        "movement_2": tf.layers.dense(h, MOVEMENT_CLASSES, name="logits_movement_2"),
        "pull": tf.layers.dense(h, BINARY_CLASSES, name="logits_pull"),
        "glueall": tf.layers.dense(h, BINARY_CLASSES, name="logits_glueall"),
    }

    losses = {
        key: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[key], logits=logits[key]))
        for key in labels
    }
    loss = tf.add_n(list(losses.values()), name="total_loss")
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    predictions = {key: tf.argmax(value, axis=1, output_type=tf.int32, name="pred_{}".format(key)) for key, value in logits.items()}
    accuracies = {
        key: tf.reduce_mean(tf.cast(tf.equal(predictions[key], labels[key]), tf.float32), name="acc_{}".format(key))
        for key in labels
    }
    movement_exact = tf.reduce_mean(tf.cast(
        tf.logical_and(
            tf.logical_and(tf.equal(predictions["movement_0"], labels["movement_0"]), tf.equal(predictions["movement_1"], labels["movement_1"])),
            tf.equal(predictions["movement_2"], labels["movement_2"]),
        ),
        tf.float32,
    ), name="movement_exact")
    action_exact = tf.reduce_mean(tf.cast(
        tf.logical_and(
            tf.logical_and(
                tf.logical_and(tf.equal(predictions["movement_0"], labels["movement_0"]), tf.equal(predictions["movement_1"], labels["movement_1"])),
                tf.equal(predictions["movement_2"], labels["movement_2"]),
            ),
            tf.logical_and(tf.equal(predictions["pull"], labels["pull"]), tf.equal(predictions["glueall"], labels["glueall"])),
        ),
        tf.float32,
    ), name="action_exact")

    return {
        "x": x,
        "is_training": is_training,
        "labels": labels,
        "loss": loss,
        "losses": losses,
        "train_op": train_op,
        "accuracies": accuracies,
        "movement_exact": movement_exact,
        "action_exact": action_exact,
    }


def feed_dict(model, x_values, labels, indices, training=False):
    feed = {
        model["x"]: x_values[indices],
        model["is_training"]: training,
    }
    for key, placeholder in model["labels"].items():
        feed[placeholder] = labels[key][indices]
    return feed


def evaluate(sess, model, x_values, labels, indices, batch_size):
    fetches = {
        "loss": model["loss"],
        "movement_exact": model["movement_exact"],
        "action_exact": model["action_exact"],
    }
    for key, tensor in model["accuracies"].items():
        fetches["acc_{}".format(key)] = tensor

    totals = {key: 0.0 for key in fetches}
    n_seen = 0
    for batch_idx in batch_iter(indices, batch_size, np.random.RandomState(0), shuffle=False):
        result = sess.run(fetches, feed_dict=feed_dict(model, x_values, labels, batch_idx, training=False))
        batch_n = len(batch_idx)
        n_seen += batch_n
        for key, value in result.items():
            totals[key] += float(value) * batch_n

    return {key: round(value / max(n_seen, 1), 6) for key, value in totals.items()}


def majority_value(values):
    counts = np.bincount(values.astype(np.int32))
    return int(np.argmax(counts))


def majority_baseline(labels, train_idx, eval_idx):
    prediction = {key: majority_value(value[train_idx]) for key, value in labels.items()}
    metrics = {}
    for key, pred in prediction.items():
        metrics["acc_{}".format(key)] = round(float(np.mean(labels[key][eval_idx] == pred)), 6)

    movement_correct = (
        (labels["movement_0"][eval_idx] == prediction["movement_0"]) &
        (labels["movement_1"][eval_idx] == prediction["movement_1"]) &
        (labels["movement_2"][eval_idx] == prediction["movement_2"])
    )
    action_correct = (
        movement_correct &
        (labels["pull"][eval_idx] == prediction["pull"]) &
        (labels["glueall"][eval_idx] == prediction["glueall"])
    )
    metrics["movement_exact"] = round(float(np.mean(movement_correct)), 6)
    metrics["action_exact"] = round(float(np.mean(action_correct)), 6)
    metrics["prediction"] = prediction
    return metrics


def write_text_report(path, report):
    lines = [
        "Behavior Cloning Report",
        "dataset: {}".format(report["dataset"]),
        "samples: {}".format(report["samples"]),
        "train_samples: {}".format(report["train_samples"]),
        "val_samples: {}".format(report["val_samples"]),
        "obs_keys: {}".format(", ".join(report["obs_keys"])),
        "hidden_sizes: {}".format(report["hidden_sizes"]),
        "",
        "Final train metrics:",
        json.dumps(report["final_train"], indent=2, sort_keys=True),
        "",
        "Final validation metrics:",
        json.dumps(report["final_val"], indent=2, sort_keys=True),
        "",
        "Majority-label validation baseline:",
        json.dumps(report["majority_baseline_val"], indent=2, sort_keys=True),
    ]
    with open(path, "w") as output_file:
        output_file.write("\n".join(lines))
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default="/workspace/runs/bc_baseline")
    parser.add_argument("--obs-key", action="append", default=[])
    parser.add_argument("--role", choices=["all", "hider", "seeker"], default="all")
    parser.add_argument("--include-role", action="store_true", default=True)
    parser.add_argument("--no-include-role", dest="include_role", action="store_false")
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    data = np.load(args.dataset, allow_pickle=True)
    manifest = data["manifest"].item()
    obs_keys = select_obs_keys(data, args.obs_key)
    x = build_features(data, obs_keys, args.include_role)
    labels = labels_from_data(data)

    mask = role_mask(data, args.role)
    train_idx, val_idx, split_info = split_indices_by_episode(data, mask, args.train_frac, args.seed)
    x, feature_mean, feature_std = normalize_features(x, train_idx)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    model = build_model(
        input_dim=x.shape[1],
        hidden_sizes=hidden_sizes,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )

    history = []
    rng = np.random.RandomState(args.seed)
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, args.epochs + 1):
            epoch_losses = []
            for batch_idx in batch_iter(train_idx, args.batch_size, rng, shuffle=True):
                _, loss_value = sess.run(
                    [model["train_op"], model["loss"]],
                    feed_dict=feed_dict(model, x, labels, batch_idx, training=True),
                )
                epoch_losses.append(float(loss_value))

            train_metrics = evaluate(sess, model, x, labels, train_idx, args.batch_size)
            val_metrics = evaluate(sess, model, x, labels, val_idx, args.batch_size)
            row = {
                "epoch": epoch,
                "mean_batch_loss": round(float(np.mean(epoch_losses)), 6),
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            print(
                "epoch={epoch} loss={loss:.4f} train_action_exact={train_action:.4f} "
                "val_action_exact={val_action:.4f} val_movement_exact={val_move:.4f}".format(
                    epoch=epoch,
                    loss=row["mean_batch_loss"],
                    train_action=train_metrics["action_exact"],
                    val_action=val_metrics["action_exact"],
                    val_move=val_metrics["movement_exact"],
                )
            )

        checkpoint_path = saver.save(sess, os.path.join(args.out_dir, "model.ckpt"))

    np.savez_compressed(
        os.path.join(args.out_dir, "preprocessing.npz"),
        feature_mean=feature_mean,
        feature_std=feature_std,
        obs_keys=np.asarray(obs_keys, dtype=object),
        include_role=np.asarray(args.include_role),
    )

    report = {
        "created_unix": time.time(),
        "dataset": args.dataset,
        "dataset_manifest": make_serializable(manifest),
        "out_dir": args.out_dir,
        "checkpoint_path": checkpoint_path,
        "preprocessing_path": os.path.join(args.out_dir, "preprocessing.npz"),
        "samples": int(len(x)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "role": args.role,
        "obs_keys": obs_keys,
        "input_dim": int(x.shape[1]),
        "hidden_sizes": hidden_sizes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "include_role": args.include_role,
        "split": split_info,
        "majority_baseline_train": majority_baseline(labels, train_idx, train_idx),
        "majority_baseline_val": majority_baseline(labels, train_idx, val_idx),
        "final_train": history[-1]["train"] if history else {},
        "final_val": history[-1]["val"] if history else {},
        "history": history,
    }

    report_path = os.path.join(args.out_dir, "metrics.json")
    with open(report_path, "w") as output_file:
        json.dump(make_serializable(report), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    write_text_report(os.path.join(args.out_dir, "metrics.txt"), report)

    print("saved_checkpoint:", checkpoint_path)
    print("saved_metrics:", report_path)
    print("saved_preprocessing:", os.path.join(args.out_dir, "preprocessing.npz"))
    print("final_val:", json.dumps(report["final_val"], sort_keys=True))


if __name__ == "__main__":
    main()
