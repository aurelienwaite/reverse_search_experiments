#!/usr/bin/env python3
import numpy as np
import numpy.typing as npt
import json
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import subprocess
from os import mkdir
import cProfile
import time
import datetime
from tenacity import retry, stop_after_attempt
from scipy import sparse
from typing import Dict, List

num_directions = 50
num_polylearn_states = 10
sample_size = 100
experiment_name = "exp2"

def split_features_labels(df: pl.DataFrame):
    labels = df["Cluster_ID", "loss"]
    labels.rename({"loss" : "Response"})
    train = df.clone().drop("Cluster_ID", "loss").with_columns(pl.lit(1, dtype=pl.Float64).alias("bias")).cast(pl.Float64)
    return train, labels

train = pl.read_parquet("data/rs_train.parquet")
train_features, train_labels = split_features_labels(train)
response = train_labels.drop("Cluster_ID").to_numpy()
train_features_np = train_features.to_numpy()
validation = pl.read_parquet("data/rs_validation.parquet")
clusters_parquet = pl.read_parquet("data/clusters.parquet")
clusters_dict = dict(zip(clusters_parquet["cluster_id"], clusters_parquet["cluster_value"]))
clusters = np.array([clusters_dict[cl_ind] for cl_ind in sorted(clusters_dict.keys())])
num_classes = len(clusters)

rs = np.random.RandomState(42)
np.random.set_state(rs.get_state())
rng = np.random.default_rng()
mkdir(experiment_name)

num_rows, num_features = (len(train), len(train.columns) - 1)
print(num_rows, num_features)
input_size = num_features

polytope_dir = f"{experiment_name}/polytope"
mkdir(polytope_dir)
def polytope_table_file(iteration):
    return f"{polytope_dir}/{iteration}.parquet"

states_dir = f"{experiment_name}/states_out"
mkdir(states_dir)
def states_file(iteration):
    return f"{states_dir}/{iteration}"

labels_dir = f"{experiment_name}/labels"
mkdir(labels_dir)
def labels_file(iteration):
    return f"{labels_dir}/{iteration}.parquet"

params_dir = f"{experiment_name}/params"
mkdir(params_dir)

"""
Worry about this later - we will need someone to compute a fast dot product over all records
print("to numpy")
num_batches = 0
num_rows = 0
for batch in features.to_arrow().to_batches():
    if num_batches % 1000:
        print(f"Done {num_batches} batches and {num_rows} rows")
    num_rows += len(batch.to_tensor().to_numpy())
    num_batches += 1
"""


def mean_average_error(test_params: npt.NDArray[np.float64], to_score: npt.NDArray[np.float64] = train_features_np, response_subset: npt.NDArray[np.float64] = response):
    reshaped_new_params = test_params.reshape((num_classes, -1))
    poly_scores = to_score @ reshaped_new_params.transpose()
    maximised = poly_scores.argmax(axis=1)
    print("Computing maximisers")
    mae = np.abs(response_subset - clusters[maximised]).mean()
    return mae

def make_projection(params: npt.NDArray[np.float64], sampled_rows: npt.NDArray[np.float64]):
    print("Making projection")
    non_zeros = {}
    for i in range(input_size):
        for image in sampled_rows:
            if i==num_features or image[i] != 0:
                non_zeros[i] = non_zeros.get(i, 0) + 1
    non_zero_keys = list(non_zeros.keys())
    print(f"There are {len(non_zero_keys)} non-zero cols")
    
    one_hots = []
    while len(one_hots) < num_directions:
        cluster = np.random.randint(0, num_classes)
        non_zero_dir = non_zero_keys[np.random.randint(0, len(non_zero_keys))]
        one_hot = non_zero_dir + cluster * input_size
        if one_hot not in one_hots:
            one_hots.append(one_hot)

    dir_shape = [len(one_hots), num_classes * input_size]
    directions = np.zeros(dir_shape, dtype="float64")
    for i, one_hot in enumerate(one_hots):
        directions[i, one_hot] = 1.0
    projection = np.concatenate([directions, params.reshape([1, -1])], axis=0)
    with open(f"{experiment_name}/dir_samples.json", "a") as one_hots_file:
        print(json.dumps(one_hots), file=one_hots_file)
    with open(f"{experiment_name}/projection_dump.json", "w") as proj_file:
        json.dump(projection.tolist(), proj_file, indent=2)
    # Because the feature vector is mostly sparse, we reshape our projection matrix
    reshaped = projection.transpose().reshape([num_classes, input_size, -1])
    print("Sampled rows shape", sampled_rows.shape)
    projected = sampled_rows @ reshaped
    return projected, reshaped

def make_samples():
    print("Making samples")
    samples = train.sample(sample_size)
    samples.write_parquet(f"{experiment_name}/samples.parquet")
    sampled_features, sampled_labels = split_features_labels(samples)
    rows_as_numpy = sampled_features.to_numpy()
    return sampled_labels, rows_as_numpy

def write_polytopes_file(projected: npt.NDArray[np.float64], labels: pl.DataFrame, iteration):
    print("Writing polytope files")
    proj_dim = projected.shape[-1]
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

    labels.rename({"loss": "Response"}).write_parquet(labels_file(iteration))


def update_params(iteration, projection):

    states: npt.NDArray[np.float64] = pl.read_parquet(states_file(iteration) + ".parquet").to_numpy()

    filtered: List[npt.NDArray[np.float64]] = [state for state in states if state[-1] > 0]

    scored = []
    for proj_param in filtered:
        new_params = proj_param.reshape((-1, 1))
        print(projection.shape, new_params.shape)
        updated = projection @ new_params
        print("Computing MAE")
        full_set_accuracy = mean_average_error(updated)
        print(full_set_accuracy)
        scored.append((full_set_accuracy, new_params))
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
            "--reverse-search-out",
            states_file(iteration),
            "--responses",
            labels_file(iteration),
            "--clusters", 
            "data/clusters.parquet",
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
    full_set_accuracy = mean_average_error(params)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    with open(f"{experiment_name}/log.txt", "a") as log_file:
        print(f"{timestamp} Finished iteration {iteration} with accuracy {full_set_accuracy}", file=log_file)

    params_file = f"{params_dir}/{iteration}.numpy"
    np.save(params_file, params)

"""try: 
    mkdir(experiment_name)
except FileExistsError:
    print("Previous experiment exists, restarting")
    with open(params_file) as params_f:
        all_params = params_f.readlines()
    start_index = len(all_params)
    params = np.array(json.loads(all_params[-1]))
    del all_params
    prev_full_set_acc = accuracy(params)"""



#@retry(stop=stop_after_attempt(10))
def do_iteration(i, params, prev_full_set_acc):
    print(f"Starting iteration {i}")
    sampled_labels, sampled_rows = make_samples()
    projected, projection = make_projection(params, sampled_rows)
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

print(f"Generating params {num_classes} by {input_size}")
params = rng.standard_normal(size=[num_classes, input_size], dtype="float64")
prev_full_set_acc = 0
start_index = 0

for i in range(start_index, 10000):
    with cProfile.Profile() as pr:
        params, prev_full_set_acc = do_iteration(i, params, prev_full_set_acc)
        pr.dump_stats(f"{experiment_name}/stats_{i}.txt")
