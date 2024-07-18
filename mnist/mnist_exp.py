#!/usr/bin/env python

from mnist import MNIST
import numpy as np
import json
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
import matplotlib.pyplot as plt
import subprocess
from os import mkdir
import cProfile
import time
import datetime


num_classes = 10
num_directions = 999
proj_dim = num_directions + 1
rs = np.random.RandomState(42)
np.random.set_state(rs.get_state())
rng = np.random.default_rng()
experiment_name = "exp7"
num_polylearn_states = 5

mndata = MNIST("/Users/rorywaite/code/polylearn/mnist_rs/data")
images, labels = mndata.load_training()
images_by_label = []
for _ in range(len(set(labels))):
    images_by_label.append([])

for image, label in zip(images, labels):
    images_by_label[label].append(image)

training_set = np.array(images)
training_labels = np.array(labels)
training_set_by_label = [np.array(ary) for ary in images_by_label]

print(training_set_by_label[0].dtype)

num_images, img_size = np.shape(training_set)
sample_size = num_images // 300

mkdir(experiment_name)


def polytope_table_file(iteration):
    return f"{experiment_name}/mnist_iter_{iteration}.parquet"

def states_file(iteration):
    return f"{experiment_name}/states_out_iter_{iteration}.jsonl"

def labels_file(iteration):
    return f"{experiment_name}/labels_iter_{iteration}.json"


def accuracy(test_params):
    reshaped_new_params = test_params.reshape((num_classes, -1))
    print(reshaped_new_params.shape, training_set.shape)
    poly_scores = training_set @ reshaped_new_params.transpose()
    maximised = poly_scores.argmax(axis=1)
    return np.equal(maximised, training_labels).sum() / len(maximised)


def make_projection(params, direction_weights):
    dir_shape = [num_directions, num_classes * img_size]
    # directions_noise = rng.standard_normal(
    #    size=dir_shape, dtype="float64"
    # )  # + np.ones(shape=dir_shape, dtype='float64')
    # directions = directions_noise  # + directions
    #copied_weights = list(direction_weights)
    #def make_bins():
    #    total_weight = sum(copied_weights)
    #    weight_bins = [copied_weights[0] / total_weight]
    #    for weight in copied_weights[1:]:
    #        weight_bins.append(weight_bins[-1] + weight / total_weight)
    #    return weight_bins
    
    #with open(f"{experiment_name}/bins.json", "a") as bins_file:
    #    print(json.dumps(make_bins()), file=bins_file)
    #one_hots = []
    #while len(one_hots) < num_directions:
    #    bins = make_bins()
    #    dir_sample = np.random.uniform()
    #    one_hot = 0
    #    while dir_sample > bins[one_hot]:
    #        one_hot += 1
    #    copied_weights.pop(one_hot)
    #    for prev_one_hot in one_hots:
    #        if one_hot >= prev_one_hot:
    #            one_hot += 1
    #    assert one_hot not in one_hots
    #    one_hots.append(one_hot)
    #    one_hots.sort()
    #with open(f"{experiment_name}/dir_samples.json", "a") as one_hots_file:
    #    print(json.dumps(one_hots), file=one_hots_file)

    #one_hots = np.random.choice(img_size * num_classes, size=num_directions, replace=False)
    directions = np.zeros(dir_shape, dtype="float64")
    one_hots = np.random.choice(img_size * num_classes, size=num_directions, replace=False)
    for i, one_hot in enumerate(one_hots):
        directions[i, one_hot] = 1.0
    projection = np.concatenate([directions, np.expand_dims(params, 0)], axis=0)
    return projection, one_hots


def make_projected(projection):
    sample_indices = np.random.choice(num_images, size=sample_size, replace=False)
    #sample_indices = list(range(num_images))
    samples = []
    for index in sample_indices:
        samples.append((labels[index], images[index]))
        # if len(samples) == num_classes:
        #    break

    sampled_labels, sampled_images = (np.array(ary) for ary in zip(*samples))
    # projected = training_set[sample_indices, :] @ reshaped
    # Because the feature vector is mostly sparse, we reshape our projection matrix
    reshaped = projection.transpose().reshape([num_classes, img_size, -1])
    projected = sampled_images @ reshaped
    return projected, sampled_labels


def write_polytopes_file(projected, labels, iteration):
    print(projected.shape)
    polytopes = projected.transpose([1, 0, 2]).reshape([-1])
    polytope_dim = num_classes * proj_dim
    print(polytope_dim)
    vertex_index = (
        np.tile(np.arange(stop=polytope_dim, dtype="i"), len(labels)) // proj_dim
    )
    polytope_index = (
        np.arange(stop=polytope_dim * len(labels), dtype="i") // polytope_dim
    )
    dim = np.tile(np.arange(stop=proj_dim, dtype="i"), len(labels) * num_classes)

    with open(labels_file(iteration), "w") as label_file:
        json.dump(labels.tolist(), label_file)

    print(len(polytope_index), len(vertex_index), len(dim), len(polytopes))
    polytope_table = pa.table(
        {
            "polytope": polytope_index,
            "vertex": vertex_index,
            "dim": dim,
            "value": polytopes,
        }
    )
    pq.write_table(polytope_table, polytope_table_file(iteration))


def update_params(iteration):

    with open(states_file(iteration), "rt") as states_file_handle:
        states = [json.loads(line) for line in states_file_handle]

    filtered = [state for state in states if state["param"]["data"][-1] > 0]

    filtered.sort(key=lambda x: x["accuracy"], reverse=True)

    scored = []
    for state in filtered:
        # print(filtered[0]["minkowski_decomp"], filtered[0]["accuracy"])
        proj_param = np.array(state["param"]["data"])

        proj_param = proj_param / proj_param[-1]

        new_params = proj_param.reshape((-1, 1)) * projection
        new_params = new_params.sum(axis=0)
        full_set_accuracy = accuracy(new_params)
        print(full_set_accuracy)
        scored.append((full_set_accuracy, state["accuracy"], new_params))
    scored.sort(key=lambda x: x[0], reverse=True)

    with open(f"{experiment_name}/scores_log.jsonl", "a") as scores_file:
        scores_file.write(
            json.dumps([{"full_set": item[0], "sample": item[1]} for item in scored]) + "\n"
        )

    best = scored[0]
    return best[0], best[2]


def run_exectuable(iteration):
    cmd = [
            "/Users/rorywaite/code/polylearn/target/release/reverse_search_main",
            "--polytope-file",
            polytope_table_file(iteration),
            "--polytope-out",
            "/tmp/deleteme.json",
            "--reserve-search-out",
            states_file(iteration),
            "--labels",
            labels_file(iteration),
            "--num-states",
            str(num_polylearn_states)
        ]
    print("Running command")
    print(" ".join(cmd))
    cp = subprocess.run(
        cmd,
        capture_output=True
    )
    print(cp.stdout.decode())
    print(cp.stderr.decode())
    cp.check_returncode()

def log_iteration(params, iteration):
    full_set_accuracy = accuracy(params)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    with open(f"{experiment_name}/log.txt", "a") as log_file:
        print(f"{timestamp} Finished iteration {iteration} with accuracy {full_set_accuracy}", file=log_file)

    with open(f"{experiment_name}/params.txt", "a") as params_file:
        params_file.write(json.dumps(params.tolist()) + "\n")

def update_direction_weights(weights, one_hots, accuracy_diff):
    for one_hot in one_hots:
        weights[one_hot] += weights[one_hot] + accuracy_diff
    return weights

params = param = rng.standard_normal(size=[num_classes * img_size], dtype="float64")
prev_full_set_acc = 0
direction_weights = [0.1] * (img_size * num_classes)

for i in range(10000):
    with cProfile.Profile() as pr:
        projection, one_hots = make_projection(params, direction_weights)
        projected, sampled_labels = make_projected(projection)
        write_polytopes_file(projected, sampled_labels, i)
        run_exectuable(i)
        full_set_accuracy, updated_params = update_params(i)
        if full_set_accuracy > prev_full_set_acc:
            params = updated_params
            if i > 0:
                direction_weights = update_direction_weights(direction_weights, one_hots, full_set_accuracy - prev_full_set_acc)
            prev_full_set_acc = full_set_accuracy
        else:
            print("params are no better than previous iteration")
        log_iteration(params, i)
        pr.dump_stats(f"{experiment_name}/stats_{i}.txt")
