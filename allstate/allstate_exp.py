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
from collections import defaultdict
from scipy.sparse import bsr_array, coo_array
from tenacity import retry, stop_after_attempt

from typing import Dict, List

num_directions = 1
#num_sparse_directions = 1
num_polylearn_states = 1
sample_size = 10000
experiment_name = "exp8"

def split_features_labels(df: pl.DataFrame):
    labels = df["Cluster_ID", "loss"]
    labels.rename({"loss" : "Response"})
    train = df.clone().drop("Cluster_ID", "loss").with_columns(pl.lit(1, dtype=pl.Float64).alias("cont_bias")).cast(pl.Float64)
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

cont_dims = {col_ind for col_ind, col_name in enumerate(train_features.columns) if col_name.startswith("cont")}

rs = np.random.RandomState(42)
np.random.set_state(rs.get_state())
rng = np.random.default_rng()
mkdir(experiment_name)

num_rows, num_features = (len(train_features), len(train_features.columns))
print(num_rows, num_features)
print(train_features_np.shape)
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

rs_logs_dir = f"{experiment_name}/rs_logs"
mkdir(rs_logs_dir)
def rs_logs_file(iteration):
    return f"{rs_logs_dir}/{iteration}.txt"

def mean_average_error(test_params: npt.NDArray[np.float64], to_score: npt.NDArray[np.float64] = train_features_np, response_subset: npt.NDArray[np.float64] = response):
    reshaped_new_params = test_params.reshape((num_classes, -1))
    poly_scores = to_score @ reshaped_new_params.transpose()
    maximiser_batch_size = 10000 # Selecting maxes takes a lot of memory
    maximised = poly_scores.argmax(axis=1)
    batches = []
    end_range = len(maximised) // maximiser_batch_size
    for batch_num in range(end_range):
        start = batch_num * maximiser_batch_size
        end = (batch_num + 1) * maximiser_batch_size
        batches.append(clusters[maximised[start:end]])
    end_index = end_range * maximiser_batch_size
    remainder = end_index + (len(maximised) % maximiser_batch_size)
    batches.append(clusters[maximised[end_index:remainder]])
    predictions = np.concatenate(batches, axis=0)
    mae = np.abs(response_subset.squeeze() - predictions).mean()
    return mae

def make_projection(params: npt.NDArray[np.float64], sampled_rows: npt.NDArray[np.float64]):
    print("Making projection")
    non_zeros = {}
    print(len(cont_dims))
    for i in range(input_size):
        for image in sampled_rows:
            if image[i] != 0:
                non_zeros[i] = non_zeros.get(i, 0) + 1
    non_zero_keys = list(non_zeros.keys())
    print(f"There are {len(non_zero_keys)} non-zero cols")
    
    directions = []
    features_to_search = np.random.choice(np.array(non_zero_keys), size=[num_directions], replace=False)
    for feature_ind in features_to_search:
        rows = []
        cols = []
        data = []
        for cluster in range(num_classes): 
            rows.append(cluster)
            cols.append(feature_ind)
            data.append(1)
            sparse_array = coo_array((np.array(data),(np.array(rows), np.array(cols))), shape=(num_classes, input_size), dtype="float64")
            directions.append(sparse_array.tobsr())

    rows_t = sampled_rows.transpose()
    projected = []
    for direction in directions:
        projected_dir = direction.dot(rows_t)
        projected.append(coo_array(projected_dir.transpose()))
    proj_params = params @ rows_t
    projected.append(proj_params.transpose())
    """with open(f"{experiment_name}/dir_samples.json", "a") as one_hots_file:
        print(json.dumps(one_hots), file=one_hots_file)
    with open(f"{experiment_name}/projection_dump.json", "w") as proj_file:
        json.dump(projection.tolist(), proj_file, indent=2)"""
    # Because the feature vector is mostly sparse, we reshape our projection matrix
    return projected, features_to_search

def make_samples():
    print("Making samples")
    samples = train.sample(sample_size)
    samples.write_parquet(f"{experiment_name}/samples.parquet")
    sampled_features, sampled_labels = split_features_labels(samples)
    rows_as_numpy = sampled_features.to_numpy()
    return sampled_labels, rows_as_numpy

def write_polytopes_file(projected: List, labels: pl.DataFrame, iteration):
    print("Writing polytope files")
    """proj_dim = input_size
    polytopes = projected.transpose([1, 0, 2]).reshape([-1])
    polytope_dim = num_classes * proj_dim
    print(polytope_dim)
    vertex_index = (
        np.tile(np.arange(stop=polytope_dim, dtype="i"), len(labels)) // proj_dim
    )
    polytope_index = (
        np.arange(stop=polytope_dim * len(labels), dtype="i") // polytope_dim
    )
    dim = np.tile(np.arange(stop=proj_dim, dtype="i"), len(labels) * num_classes)"""
    polytope_index = []
    vertex_index = []
    dim_index = []
    value = []

    print(projected[0].shape)
    print(len(projected))
    expanded_directions = len(projected) - 1

    dims_by_polytope = defaultdict(lambda: set())

    for dim, direction in enumerate(projected[:-1]):
        for polytope, vertex, datum in zip(direction.row, direction.col, direction.data):
            polytope_index.append(polytope)
            vertex_index.append(vertex)
            dim_index.append(dim)
            dims_by_polytope[polytope].add(dim)
            value.append(datum)

    param_dim = expanded_directions * num_classes + 1
    for polytope in range(sample_size):
        for vertex in range(num_classes):
            polytope_index.append(polytope)
            vertex_index.append(vertex)
            dim_index.append(param_dim)
            dims_by_polytope[polytope].add(dim)
            value.append(projected[-1][polytope, vertex])

    for p_ind in sorted(dims_by_polytope.keys()):
        print(f"{p_ind}: {len(dims_by_polytope[p_ind])}")


    print(len(polytope_index), len(vertex_index), len(dim_index), len(polytope_index))
    schema = pa.schema([
        pa.field("polytope", pa.int32()),
        pa.field("vertex", pa.int32()),
        pa.field("dim", pa.int32()),
        pa.field("value", pa.float64())])
    polytope_table = pa.Table.from_pydict(
        {
            "polytope": polytope_index,
            "vertex": vertex_index,
            "dim": dim_index,
            "value": value,
        },
        schema=schema
    )
    pq.write_table(polytope_table, polytope_table_file(iteration))

    labels.rename({"loss": "Response"}).write_parquet(labels_file(iteration))


def update_params(iteration, params: npt.NDArray[np.float64], projection: List[int]):

    states: npt.NDArray[np.float64] = pl.read_parquet(states_file(iteration) + ".parquet").to_numpy()

    filtered: List[npt.NDArray[np.float64]] = [state / state[-1] for state in states if state[-1] > 0]

    scored = []
    for proj_param in filtered:
        updated = params.copy()
        assert (len(proj_param) == len(projection) + 1), f"{len(proj_param)} vs {len(projection)}"
        for axis, param in zip(projection, proj_param):
            row = axis // input_size
            col = axis % input_size
            updated[row, col] += param

        print("Computing MAE")
        full_set_accuracy = mean_average_error(updated)
        print(full_set_accuracy)
        scored.append((full_set_accuracy, updated))
    scored.sort(key=lambda x: x[0])

    with open(f"{experiment_name}/scores_log.jsonl", "a") as scores_file:
        scores_file.write(
            json.dumps([{"full_set": item[0]} for item in scored]) + "\n"
        )

    best = scored[0]
    print(f"Best score {best[0]}")
    return best[0], best[1]


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
        capture_output=True,
        #timeout = 60 * 10
    )
    logs = cp.stdout.decode() + cp.stderr.decode()
    print(logs)
    with open(rs_logs_file(iteration), "wt") as log_file:
        print(" ".join(cmd), file=log_file)
        print(logs, file=log_file)
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



@retry(stop=stop_after_attempt(100))
def do_iteration(i, params, prev_full_set_acc):
    print(f"Starting iteration {i}")
    sampled_labels, sampled_rows = make_samples()
    projected, projection = make_projection(params, sampled_rows)
    write_polytopes_file(projected, sampled_labels, i)
    run_exectuable(i)
    full_set_accuracy, updated_params = update_params(i, params, projection)
    if full_set_accuracy < prev_full_set_acc:
        params = updated_params
        prev_full_set_acc = full_set_accuracy
    else:
        print("params are no better than previous iteration")
    log_iteration(params, i)
    return params, prev_full_set_acc

params = np.load("data/ridge_params.numpy.npy")
prev_full_set_acc = mean_average_error(params)
print(f"initial params have a score of {prev_full_set_acc}")
start_index = 0

for i in range(start_index, 10000):
    with cProfile.Profile() as pr:
        params, prev_full_set_acc = do_iteration(i, params, prev_full_set_acc)
        #pr.dump_stats(f"{experiment_name}/stats_{i}.txt")
