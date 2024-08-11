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
from typing import Dict, List

num_directions = 199
#num_sparse_directions = 1
num_polylearn_states = 1
sample_size = 1000
experiment_name = "exp4"

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
with open("data/cluster_quintiles.json") as q_file:
    cluster_quantiles = json.load(q_file)["quantile_clusters"]
num_classes = len(cluster_quantiles)
inverse_quantiles = {int(cluster): int(quantile) for quantile, clusters in cluster_quantiles.items() for cluster in clusters}
quantile_labels = train_labels["Cluster_ID"].replace_strict(inverse_quantiles).to_numpy()

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

def accuracy(test_params: npt.NDArray[np.float64]):
    reshaped_new_params = test_params.reshape((num_classes, -1))
    poly_scores = train_features_np @ reshaped_new_params.transpose()
    maximised = poly_scores.argmax(axis=1)
    return np.equal(maximised, quantile_labels).sum() / len(maximised)


def make_projection(params: npt.NDArray[np.float64], sampled_rows: npt.NDArray[np.float64]):
    print("Making projection")
    non_zeros = {}
    print(len(cont_dims))
    for i in range(num_features):
        for image in sampled_rows:
            if image[i] != 0:
                non_zeros[i] = non_zeros.get(i, 0) + 1
    non_zero_keys = list(non_zeros.keys())
    print(f"There are {len(non_zero_keys)} non-zero cols")
    
    one_hots = []
    added_dims = 0
    sparse_count = 0
    cont_count = 0
    while added_dims < num_directions:
        non_zero_dir = non_zero_keys[np.random.randint(0, len(non_zero_keys))]
        if False:
            if non_zero_dir not in one_hots:
                added_dims += 1
                sparse_count += 1
                for cluster in range(num_classes): 
                    one_hot = non_zero_dir + cluster * input_size
                    if one_hot not in one_hots:
                        one_hots.append(one_hot)
        else:
            cluster = np.random.randint(0, num_classes)
            one_hot = non_zero_dir + cluster * input_size
            if one_hot not in one_hots:
                added_dims += 1
                cont_count +=1
                one_hots.append(one_hot)

    print(f"Added {added_dims} directions with {sparse_count} sparse and {cont_count} continious")
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
    quantiled_labels = sampled_labels["Cluster_ID"].replace_strict(inverse_quantiles)
    rows_as_numpy = sampled_features.to_numpy()
    return quantiled_labels, rows_as_numpy

def write_polytopes_file(projected: npt.NDArray[np.float64], labels: pl.Series, iteration):
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

    #labels.rename({"loss": "Response"}).write_parquet(labels_file(iteration))
    with open(labels_file(iteration), "w") as label_file:
        json.dump(labels.to_list(), label_file)


def update_params(iteration, projection):

    states: npt.NDArray[np.float64] = pl.read_parquet(states_file(iteration) + ".parquet").to_numpy()

    filtered: List[npt.NDArray[np.float64]] = [state for state in states if state[-1] > 0]

    scored = []
    for proj_param in filtered:
        new_params = proj_param.reshape((-1, 1))
        print(projection.shape, new_params.shape)
        updated = projection @ new_params
        print("Computing MAE")
        full_set_accuracy = accuracy(updated)
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
            "--labels",
            labels_file(iteration),
            #"--clusters", 
            #"data/clusters.parquet",
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



@retry(stop=stop_after_attempt(10))
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

print(f"Generating params {num_classes} by {len(cont_dims)}")
dense_params = rng.standard_normal(size=[num_classes, len(cont_dims)], dtype="float64")
params = np.concatenate([0.1 * np.ones([num_classes, input_size - len(cont_dims)], dtype="float64"), dense_params], axis=1)
prev_full_set_acc = 0.
start_index = 0

for i in range(start_index, 10000):
    with cProfile.Profile() as pr:
        params, prev_full_set_acc = do_iteration(i, params, prev_full_set_acc)
        pr.dump_stats(f"{experiment_name}/stats_{i}.txt")
