import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf


MOVEMENT_CLASSES = 11
BINARY_CLASSES = 2
FLOAT_OBS_KEYS = [
    "observation_self",
    "lidar",
    "agent_qpos_qvel",
    "box_obs",
    "ramp_obs",
    "mask_ab_obs_spoof",
]
MASK_OBS_KEYS = ["mask_aa_obs", "mask_ab_obs", "mask_ar_obs"]
ALL_OBS_KEYS = FLOAT_OBS_KEYS + MASK_OBS_KEYS


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


def labels_from_data(data):
    movement = data["action_action_movement"].astype(np.int32)
    return {
        "movement_0": movement[:, 0],
        "movement_1": movement[:, 1],
        "movement_2": movement[:, 2],
        "pull": data["action_action_pull"].astype(np.int32).reshape(-1),
        "glueall": data["action_action_glueall"].astype(np.int32).reshape(-1),
    }


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


def normalize_float_obs(data, train_idx):
    arrays = {}
    stats = {}
    for key in FLOAT_OBS_KEYS:
        npz_key = "obs_{}".format(key)
        value = data[npz_key].astype(np.float32)
        mean = value[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
        std = value[train_idx].std(axis=0, keepdims=True).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        arrays[key] = (value - mean) / std
        stats["mean_{}".format(key)] = mean
        stats["std_{}".format(key)] = std

    for key in MASK_OBS_KEYS:
        arrays[key] = data["obs_{}".format(key)].astype(np.float32)
    return arrays, stats


def batch_iter(indices, batch_size, rng, shuffle=True):
    indices = indices.copy()
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def dense_stack(x, sizes, scope, dropout, is_training):
    h = x
    with tf.variable_scope(scope):
        for idx, size in enumerate(sizes):
            h = tf.layers.dense(h, size, activation=tf.nn.relu, name="dense_{}".format(idx))
            if dropout > 0.0:
                h = tf.layers.dropout(h, rate=dropout, training=is_training, name="dropout_{}".format(idx))
    return h


def masked_mean(values, mask):
    mask = tf.cast(mask, tf.float32)
    while len(mask.shape) < len(values.shape):
        mask = tf.expand_dims(mask, -1)
    mask_values = tf.ones_like(values) * mask
    masked = values * mask_values
    denom = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
    return tf.reduce_sum(masked, axis=1) / denom


def masked_max(values, mask):
    mask = tf.cast(mask, tf.float32)
    while len(mask.shape) < len(values.shape):
        mask = tf.expand_dims(mask, -1)
    mask_values = tf.ones_like(values) * mask
    very_negative = tf.ones_like(values) * -1e9
    masked = tf.where(tf.equal(mask_values, 1.0), values, very_negative)
    max_value = tf.reduce_max(masked, axis=1)
    has_any = tf.cast(tf.reduce_sum(mask, axis=1) > 0.0, tf.float32)
    return max_value * (tf.ones_like(max_value) * has_any)


def entity_encoder(values, mask, hidden_size, scope):
    with tf.variable_scope(scope):
        encoded = tf.layers.dense(values, hidden_size, activation=tf.nn.relu, name="entity_dense_0")
        encoded = tf.layers.dense(encoded, hidden_size, activation=tf.nn.relu, name="entity_dense_1")
        return tf.concat([masked_mean(encoded, mask), masked_max(encoded, mask)], axis=1)


def one_hot_role(role):
    return tf.one_hot(role, depth=2, dtype=tf.float32)


def class_weights(labels, indices, n_classes):
    values = labels[indices].astype(np.int32)
    counts = np.bincount(values, minlength=n_classes).astype(np.float32)
    weights = float(len(values)) / np.maximum(counts * float(n_classes), 1.0)
    return np.clip(weights, 0.25, 8.0).astype(np.float32)


def build_model(shapes, hidden_sizes, entity_hidden, learning_rate, dropout, use_class_balance, class_weight_values):
    tf.reset_default_graph()
    inputs = {
        key: tf.placeholder(tf.float32, shape=[None] + list(shape), name="input_{}".format(key))
        for key, shape in shapes.items()
    }
    role = tf.placeholder(tf.int32, shape=[None], name="role")
    is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
    labels = {
        "movement_0": tf.placeholder(tf.int32, shape=[None], name="label_movement_0"),
        "movement_1": tf.placeholder(tf.int32, shape=[None], name="label_movement_1"),
        "movement_2": tf.placeholder(tf.int32, shape=[None], name="label_movement_2"),
        "pull": tf.placeholder(tf.int32, shape=[None], name="label_pull"),
        "glueall": tf.placeholder(tf.int32, shape=[None], name="label_glueall"),
    }

    self_h = dense_stack(inputs["observation_self"], [64], "self_encoder", dropout, is_training)
    lidar_flat_dim = int(np.prod(shapes["lidar"]))
    lidar_flat = tf.reshape(inputs["lidar"], [-1, lidar_flat_dim])
    lidar_h = dense_stack(lidar_flat, [64], "lidar_encoder", dropout, is_training)
    agent_h = entity_encoder(inputs["agent_qpos_qvel"], inputs["mask_aa_obs"], entity_hidden, "agent_encoder")

    spoof = tf.expand_dims(inputs["mask_ab_obs_spoof"], -1)
    box_input = tf.concat([inputs["box_obs"], spoof], axis=2)
    box_h = entity_encoder(box_input, inputs["mask_ab_obs"], entity_hidden, "box_encoder")
    ramp_h = entity_encoder(inputs["ramp_obs"], inputs["mask_ar_obs"], entity_hidden, "ramp_encoder")

    trunk_input = tf.concat([self_h, lidar_h, agent_h, box_h, ramp_h, one_hot_role(role)], axis=1, name="structured_features")
    trunk = dense_stack(trunk_input, hidden_sizes, "trunk", dropout, is_training)

    logits = {}
    for key, n_classes in [
        ("movement_0", MOVEMENT_CLASSES),
        ("movement_1", MOVEMENT_CLASSES),
        ("movement_2", MOVEMENT_CLASSES),
        ("pull", BINARY_CLASSES),
        ("glueall", BINARY_CLASSES),
    ]:
        with tf.variable_scope("role_heads_{}".format(key)):
            hider_logits = tf.layers.dense(trunk, n_classes, name="hider")
            seeker_logits = tf.layers.dense(trunk, n_classes, name="seeker")
            selector = tf.expand_dims(tf.cast(tf.equal(role, 0), tf.float32), 1)
            logits[key] = selector * hider_logits + (1.0 - selector) * seeker_logits

    losses = {}
    for key, value in logits.items():
        raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[key], logits=value)
        if use_class_balance:
            weights = tf.constant(class_weight_values[key], dtype=tf.float32, name="class_weights_{}".format(key))
            sample_weights = tf.gather(weights, labels[key])
            losses[key] = tf.reduce_sum(raw_loss * sample_weights) / tf.maximum(tf.reduce_sum(sample_weights), 1.0)
        else:
            losses[key] = tf.reduce_mean(raw_loss)

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
        "inputs": inputs,
        "role": role,
        "is_training": is_training,
        "labels": labels,
        "loss": loss,
        "train_op": train_op,
        "accuracies": accuracies,
        "movement_exact": movement_exact,
        "action_exact": action_exact,
    }


def feed_dict(model, arrays, labels, role_values, indices, training=False):
    feed = {model["role"]: role_values[indices], model["is_training"]: training}
    for key, placeholder in model["inputs"].items():
        feed[placeholder] = arrays[key][indices]
    for key, placeholder in model["labels"].items():
        feed[placeholder] = labels[key][indices]
    return feed


def evaluate(sess, model, arrays, labels, role_values, indices, batch_size):
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
        result = sess.run(fetches, feed_dict=feed_dict(model, arrays, labels, role_values, batch_idx, training=False))
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
        "Structured Behavior Cloning Report",
        "dataset: {}".format(report["dataset"]),
        "samples: {}".format(report["samples"]),
        "train_samples: {}".format(report["train_samples"]),
        "val_samples: {}".format(report["val_samples"]),
        "architecture: structured entity encoders + role-specific heads",
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
    parser.add_argument("--out-dir", default="/workspace/runs/bc_structured")
    parser.add_argument("--role", choices=["all", "hider", "seeker"], default="all")
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--entity-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-class-balance", dest="class_balance", action="store_false")
    parser.set_defaults(class_balance=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    data = np.load(args.dataset, allow_pickle=True)
    manifest = data["manifest"].item()
    labels = labels_from_data(data)
    mask = role_mask(data, args.role)
    train_idx, val_idx, split_info = split_indices_by_episode(data, mask, args.train_frac, args.seed)
    arrays, preprocessing = normalize_float_obs(data, train_idx)
    role_values = data["role"].astype(np.int32)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    shapes = {key: arrays[key].shape[1:] for key in ALL_OBS_KEYS}

    class_weight_values = {
        "movement_0": class_weights(labels["movement_0"], train_idx, MOVEMENT_CLASSES),
        "movement_1": class_weights(labels["movement_1"], train_idx, MOVEMENT_CLASSES),
        "movement_2": class_weights(labels["movement_2"], train_idx, MOVEMENT_CLASSES),
        "pull": class_weights(labels["pull"], train_idx, BINARY_CLASSES),
        "glueall": class_weights(labels["glueall"], train_idx, BINARY_CLASSES),
    }

    model = build_model(
        shapes=shapes,
        hidden_sizes=hidden_sizes,
        entity_hidden=args.entity_hidden,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        use_class_balance=args.class_balance,
        class_weight_values=class_weight_values,
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
                    feed_dict=feed_dict(model, arrays, labels, role_values, batch_idx, training=True),
                )
                epoch_losses.append(float(loss_value))

            train_metrics = evaluate(sess, model, arrays, labels, role_values, train_idx, args.batch_size)
            val_metrics = evaluate(sess, model, arrays, labels, role_values, val_idx, args.batch_size)
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
        model_type=np.asarray("structured_bc_v1", dtype=object),
        obs_keys=np.asarray(ALL_OBS_KEYS, dtype=object),
        float_obs_keys=np.asarray(FLOAT_OBS_KEYS, dtype=object),
        mask_obs_keys=np.asarray(MASK_OBS_KEYS, dtype=object),
        class_balance=np.asarray(args.class_balance),
        **preprocessing
    )

    report = {
        "created_unix": time.time(),
        "dataset": args.dataset,
        "dataset_manifest": make_serializable(manifest),
        "out_dir": args.out_dir,
        "checkpoint_path": checkpoint_path,
        "preprocessing_path": os.path.join(args.out_dir, "preprocessing.npz"),
        "samples": int(len(role_values)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "role": args.role,
        "architecture": "structured_entity_encoders_role_specific_heads",
        "hidden_sizes": hidden_sizes,
        "entity_hidden": args.entity_hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "class_balance": args.class_balance,
        "split": split_info,
        "class_weights": make_serializable(class_weight_values),
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
