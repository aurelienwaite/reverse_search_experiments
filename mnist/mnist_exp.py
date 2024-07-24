#!/usr/bin/env python3

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
from tenacity import retry, stop_after_attempt

num_classes = 10
num_directions = 9
proj_dim = num_directions + 1
rs = np.random.RandomState(42)
np.random.set_state(rs.get_state())
rng = np.random.default_rng()
experiment_name = "exp11"
params_file = f"{experiment_name}/params.txt"
num_polylearn_states = 1

mndata = MNIST("../../mnist_rs/data")
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
sample_size = num_images // 1


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


def make_projection(params, sampled_images, sampled_labels):
    dir_shape = [num_directions, num_classes * img_size]

    non_zeros = {}
    for i in range(img_size):
        for image, label in zip(sampled_images, sampled_labels):
            if image[i] != 0:
                by_label = non_zeros.get(label, {})
                non_zeros[label] = by_label
                by_label[i] = by_label.get(i, 0) + 1
    
    def make_weighted(label):
        items = [(k,v) for k, v in non_zeros[label].items()]
        weighted = [items[0][1]]
        for item in items[1:]:
            weighted.append(weighted[-1] + item[1])
        total_count = weighted[-1]
        weighted = [w/total_count for w in weighted]
        return weighted, [item[0] for item in items]
    
    one_hots = []
    while len(one_hots) < num_directions:
        img_class = np.random.randint(0, num_classes)
        weights, mappings = make_weighted(img_class)
        dir_sample = np.random.uniform()
        for i, weight in enumerate(weights):
            if dir_sample < weight:
                one_hot = mappings[i] + img_class * img_size
                if one_hot not in one_hots:
                    one_hots.append(one_hot)
                break

                
               
    """     subspace_img_size = len(non_zeros)
    subspace_one_hots = np.random.choice(subspace_img_size * num_classes, size=num_directions, replace=False).tolist()
    one_hots = [non_zeros[i % subspace_img_size] + (i // subspace_img_size) * img_size for i in subspace_one_hots]
    print(subspace_img_size, subspace_img_size / img_size, non_zeros[210])
    print(non_zeros)
    print(subspace_one_hots, one_hots, [non_zeros[i % num_classes] for i in subspace_one_hots]) """

    """for one_hot in one_hots:
        count = 0
        for img in sampled_images:
            if img[one_hot % img_size] != 0:
                count += 1
        print (one_hot % img_size, count)"""

    directions = np.zeros(dir_shape, dtype="float64")
    for i, one_hot in enumerate(one_hots):
        directions[i, one_hot] = 1.0
    projection = np.concatenate([directions, np.expand_dims(params, 0)], axis=0)
    with open(f"{experiment_name}/dir_samples.json", "a") as one_hots_file:
        print(json.dumps(one_hots), file=one_hots_file)
    with open(f"{experiment_name}/projection_dump.json", "w") as proj_file:
        json.dump(projection.tolist(), proj_file, indent=2)
    return projection

def make_samples():
    sample_indices = np.random.choice(num_images, size=sample_size, replace=False)
    samples = []
    for index in sample_indices:
        samples.append((labels[index], images[index]))

    sampled_labels, sampled_images = (np.array(ary) for ary in zip(*samples))
    return sampled_labels, sampled_images

def make_projected(projection, sampled_images):
    # projected = training_set[sample_indices, :] @ reshaped
    # Because the feature vector is mostly sparse, we reshape our projection matrix
    reshaped = projection.transpose().reshape([num_classes, img_size, -1])
    projected = sampled_images @ reshaped
    return projected


def write_polytopes_file(projected, labels, iteration):
    print(projected.shape)
    with open(f"{experiment_name}/projected_dump.json", "w") as dump:
        json.dump(projected.tolist(), dump, indent=2)
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


def update_params(iteration, projection):

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
            "../../target/release/reverse_search_main",
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

    with open(params_file, "a") as params_file_obj:
        params_file_obj.write(json.dumps(params.tolist()) + "\n")


params = param = rng.standard_normal(size=[num_classes * img_size], dtype="float64")
prev_full_set_acc = 0
start_index = 0
try: 
    mkdir(experiment_name)
except FileExistsError:
    print("Previous experiment exists, restarting")
    with open(params_file) as params_f:
        all_params = params_f.readlines()
    start_index = len(all_params)
    params = np.array(json.loads(all_params[-1]))
    del all_params
    prev_full_set_acc = accuracy(params)


@retry(stop=stop_after_attempt(10))
def do_iteration(i, params, prev_full_set_acc):
    sampled_labels, sampled_images = make_samples()
    projection = make_projection(params, sampled_images, sampled_labels)
    projected = make_projected(projection, sampled_images)
    write_polytopes_file(projected, sampled_labels, i)
    run_exectuable(i)
    full_set_accuracy, updated_params = update_params(i, projection)
    if full_set_accuracy > prev_full_set_acc:
        params = updated_params
        prev_full_set_acc = full_set_accuracy
    else:
        print("params are no better than previous iteration")
    log_iteration(params, i)
    return params, prev_full_set_acc

for i in range(start_index, 10000):
    with cProfile.Profile() as pr:
        params, prev_full_set_acc = do_iteration(i, params, prev_full_set_acc)
        pr.dump_stats(f"{experiment_name}/stats_{i}.txt")
