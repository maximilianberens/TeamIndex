from TeamIndex import creation, evaluation

import string
from pathlib import Path, PosixPath
import os
import glob

from getpass import getpass, getuser
from subprocess import Popen, PIPE
from typing import List, Optional
import socket

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
import random
import numpy as np
import pandas as pd
from pyarrow import parquet as pap

import yaml
import json

import re



def collect_experiment_files(root_folder):
    # Match timestamp like 2025_04_14_16_06_34
    TIMESTAMP_PATTERN = r"\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}"
    config_folder = Path(root_folder) / "configs"
    result_folder = Path(root_folder) / "results"
    config_files = sorted(config_folder.glob("query-*.json"))

    experiments = []

    for config_path in config_files:
        with open(config_path, "r") as f:
            config = json.load(f)

        executor_config = config.get("executor_config", {})
        experiment_name = executor_config.get("experiment_name")
        task_stats_pattern = executor_config.get("print_task_stats")
        result_stats_pattern = executor_config.get("print_result_stats")

        if not experiment_name or not task_stats_pattern or not result_stats_pattern:
            continue  # Skip malformed config

        task_stats_regex = re.compile(rf"^task_stats-{re.escape(experiment_name)}-({TIMESTAMP_PATTERN})\.json$")
        # result_stats_regex = re.compile(rf"^result_stats_{re.escape(experiment_name)}_({TIMESTAMP_PATTERN})\.json$")

        all_files = list(result_folder.glob("*.json"))

        for file in all_files:
            match = task_stats_regex.match(file.name)
            if not match:
                continue
            timestamp = match.group(1)
            expected_result_name = f"result_stats-{experiment_name}-{timestamp}.json"
            expected_result_path = result_folder / expected_result_name
            if expected_result_path.exists():
                experiments.append({
                    "config_path": config_path,
                    "experiment_name": experiment_name,
                    "task_stats_file": file,
                    "result_stats_file": expected_result_path
                })
            else:
                print(f"ERROR: Expected result stats file not found: {expected_result_path}!!")

    return experiments

def parse_task_stats(path):
    run_stats = dict()

    task_data, meta = import_benchmark_data(path)

    ## aggregated task times per worker, in ms
    aggregated_task_stats = plot_task_stats_aggregated(task_data, meta, output_path = None)

    total_runtime_ns = meta.get('execution_time_ns', 0)
    theoretical_available_time_ms = aggregated_task_stats.shape[0] * total_runtime_ns / 1_000_000
    total_task_time_ms = aggregated_task_stats.sum().sum()
    worker_utilization = total_task_time_ms / theoretical_available_time_ms

    run_stats["worker_utilization"] = round(worker_utilization, 3)
    # run_stats["execution_time_ms"] = exec_time
    run_stats["worker_count"] = aggregated_task_stats.shape[0]

    # Calculate the total time spent on each task type
    sum_work = aggregated_task_stats.sum()  # gives total work per task type in ms
    sum_work = sum_work / sum_work.sum()
    run_stats["most_expensive_task"] = sum_work.idxmax()
    run_stats["most_expensive_task_rel_work"] = round(float(sum_work.max()/sum_work.sum()), 3)
    
    # count tasks that were actually run (i.e., have a start time != None)
    task_counts = task_data.groupby(['worker_id', 'type']).size().unstack(fill_value=0)
    run_stats["executed_task_count"] = task_counts.sum().sum()

    return run_stats

def parse_query_spec(path):
    query_specs = dict()

    spec = creation.open_json(path)

    query_specs["experiment_name"] = spec["executor_config"]["experiment_name"]  # e.g. "q0-roaring-liburing-expand_first_Team-idx1"
    if "variant_name" in spec["executor_config"].keys():
        query_specs["variant_name"] = spec["executor_config"]["variant_name"]
    else:
        ## try to parse it from the experiment name... (temporary workaround)
        query_specs["variant_name"] = query_specs["experiment_name"].split("-")[3]  # e.g., "expand_first_Team"
    query_specs["worker_count"] = spec["executor_config"]["worker_count"]
    
    query_specs["backend"] = spec["executor_config"]["backend"]
    if spec["executor_config"]["backend"] == "liburing":
        query_specs["queue_pair_count"] = spec["storage_config"]["queue_pair_count"]
        query_specs["queue_depth"] = spec["storage_config"]["liburing_cfg"]["queue_depth"]
    else:
        query_specs["queue_pair_count"] = None
        query_specs["queue_depth"] = None
    
    query_specs["leaf_union_list_parallel_threshold"] = spec["plan_config"]["leaf_union_list_parallel_threshold"]
    if "distributed_intersection_parallel_threshold" in spec["plan_config"]:
        query_specs["distributed_intersection_parallel_threshold"] = spec["plan_config"]["distributed_intersection_parallel_threshold"]
    else:
        query_specs["distributed_intersection_parallel_threshold"] = None
    query_specs["ise_count"] = spec["global_info"]["ise_count"]
    query_specs["table_cardinality"] = spec["global_info"]["table_cardinality"]
    query_specs["total_input_cardinality"] = spec["global_info"]["total_input_cardinality"]
    query_specs["total_read_volume_GB"] = spec["global_info"]["total_read_volume_KiB"]*1024/1000/1000/1000
    query_specs["total_compressed_size_KB"] = spec["global_info"]["total_compressed_size_KB"]
    query_specs["total_request_count"] = spec["global_info"]["total_request_count"]

    # calculate uncompressed size from the number of ids:
    query_specs["query_uncompressed_size_KB"] = query_specs["total_input_cardinality"] * 4 / 1000
    # calculate read amplification
    query_specs["query_read_amplification"] = round(spec["global_info"]["total_read_volume_KiB"]*1024/1000 / query_specs["total_compressed_size_KB"], 4)
    # calculate compression ratio
    query_specs["query_compression_ratio_uc-c"] = round(query_specs["query_uncompressed_size_KB"] / query_specs["total_compressed_size_KB"], 4)
    # total request count
    query_specs["query_total_request_count"] = spec["global_info"]["total_request_count"]

    # create a hash for the query string under "query" to have a compact group key for arbitrary queries
    query_specs["query_hash"] = creation.string_to_hash(spec["query"], 8)

    query_specs["index_id"] = int(spec["index_id"])
    query_specs["compression"] = spec["global_info"]["compression"]
    
    ## aggregate some information for the Teams
    included_count = 0
    ise_length = 0
    total_group_count = 0
    list_count = 0
    for team_dict in spec["team_workload_infos"]:
        if not team_dict["expand"]:
            ise_length += team_dict["group_count"]
        if team_dict["is_included"]:
            included_count += 1
        total_group_count += team_dict["group_count"]
        list_count += team_dict["list_cnt"]

    query_specs["first_team_cardinality"] = spec["team_workload_infos"][0]["total_cardinality"]
    query_specs["first_team_comp_volume"] = spec["team_workload_infos"][0]["total_size_comp"]
    query_specs["included_count"] = included_count
    query_specs["ise_length"] = ise_length
    query_specs["total_group_count"] = total_group_count
    query_specs["list_count"] = list_count

    return query_specs

def parse_result_stats(path):
    runtime_cfg = creation.open_json(path)
    result_stats = dict()
    result_stats["executor_runtime_ms"] = runtime_cfg["executor_runtime"] / 1e6  # in milliseconds
    result_stats["plan_construction_runtime_ms"] = runtime_cfg["plan_construction_runtime"] / 1e6  # in milliseconds
    result_stats["team_count"] = runtime_cfg["team_count"]
    result_stats["expanded_team_count"] = runtime_cfg["expanded_team_count"]
    result_stats["result_cardinality"] = runtime_cfg["result_cardinality"]
    result_stats["table_cardinality"] = runtime_cfg["table_cardinality"]
    result_stats["total_list_size_uncompressed"] = runtime_cfg["table_cardinality"] * result_stats["team_count"]
    

    return result_stats

def get_team_stats(query_spec_path, index_def_glob="index*.json"):
    spec = creation.open_json(query_spec_path)
    team_folder = Path(spec["team_workload_infos"][0]["team_file_path"]).parent
    assert team_folder.exists(), f"Parent folder of Team's list file does not exist: {team_folder}"

    matched_files = sorted(team_folder.glob(index_def_glob))

    assert len(matched_files), f"No index definitions matching \"{index_def_glob}\" found in folder {team_folder}!"

    if len(matched_files) > 1:
        print("Found multiple index definitions..picking first!")

    index_def_path = matched_files[0]

    team_stats = dict()
    compression = "roaring"

    if "compression" in spec["global_info"]:
        compression = spec["global_info"]["compression"]

    team_index = evaluation.TeamIndex(index_def_path, compression=compression)

    team_stats["total_list_storage_costs_GB"] = sum(list(team_index.stats['index_size_on_disk'].values()))/1000/1000/1000
    team_stats["total_padding_GB"] = sum(list(team_index.stats['total_padding'].values()))/1000/1000/1000
    team_stats["total_meta_data_size_MB"] = sum(list(team_index.stats['index_in_memory_structure_sizes'].values()))/1000/1000
    
    netto_compressed_list_size_byte = sum(list(team_index.stats['compressed_size'].values()))
    team_count = len(team_index.stats['team_shapes'])
    netto_uncompressed_list_size_byte = team_index.id_type().nbytes*team_index.stats['number_of_tuples']*team_count
    team_stats["total_uncompressed_list_size_GB"] = netto_uncompressed_list_size_byte/1000/1000/1000
    team_stats["compression_ratio_uc-c"] = round(netto_uncompressed_list_size_byte/netto_compressed_list_size_byte, 4)
    unknown_compression_codec_usage_count = sum([stat.get(evaluation.CodecID(0), 0) for stat in team_index.stats['codec_usage'].values()])
    total_bin_count = sum([sum(usage for usage in stat.values()) for stat in team_index.stats['codec_usage'].values()])
    team_stats["empty_bins_ratio"] = round(unknown_compression_codec_usage_count/total_bin_count, 3)

    non_zero_bin_count = sum([sum(usage for codec, usage in stat.items() if codec != evaluation.CodecID(0))
                              for stat in team_index.stats['codec_usage'].values()])
    copy_compression_codec_usage_count = sum([stat.get(evaluation.CodecID(1), 0)
                                              for stat in team_index.stats['codec_usage'].values()])
    team_stats["copy_codec_usage_ratio"] = round(copy_compression_codec_usage_count/non_zero_bin_count, 3)


    team_stats["d"] = max(len(sh) for sh in team_index.stats["team_shapes"].values())
    team_stats["b"] = max(max(sh) for sh in team_index.stats["team_shapes"].values())
    team_stats["team_count"] = team_count
    
    return team_stats

def build_experiment_dataframe(root_folder, drop_fails=True):
    experiments = collect_experiment_files(root_folder)
    records = []
    run_counters = {}

    for exp in experiments:
        name = exp["experiment_name"]
        run_id = run_counters.get(name, 0)

        row = {
            "run_id": run_id,
            "experiment_name": name,
            "query_spec_file": str(exp["config_path"]),
            "task_stats_file": str(exp["task_stats_file"]),
            "result_stats_file": str(exp["result_stats_file"])
        }

        # Merge in parsed stats
        row.update(parse_query_spec(exp["config_path"]))
        row.update(get_team_stats(exp["config_path"]))
        row.update(parse_task_stats(exp["task_stats_file"]))
        row.update(parse_result_stats(exp["result_stats_file"]))

        records.append(row)
        run_counters[name] = run_id + 1

    if len(records) == 0:
        print("No experiments found in the specified folder.")
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    
    cols = set(df.columns)
    # Only attributes that are run-time statistics (or may vary from run to run) should be regular columns
    non_index_cols = {"executor_runtime_ms", "plan_construction_runtime_ms", "worker_utilization", "result_cardinality",
                      "task_stats_file", "result_stats_file"}
    index_cols = list(cols - non_index_cols)
    
    potential_fails = df.query("result_cardinality == 0")
    if potential_fails.shape[0]:
        print(f"Warning: {potential_fails.shape[0]} experiments had 0 ids as result!")
    if drop_fails:
        df.query("result_cardinality != 0", inplace=True)

    df.set_index(index_cols, inplace=True)
    df.sort_index(inplace=True)
    return df.reset_index(drop=False)


def import_benchmark_data(file_path):
    """
    Imports benchmark data from a JSON file or the latest JSON file in a folder
    into a pandas DataFrame.

    Args:
        path (str): The path to either a JSON file or a folder containing JSON files.
                     If it's a folder, the function will find, sort, and open the
                     latest JSON file based on name.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: DataFrame containing the task-specific statistics.
            - dict: Dictionary containing any other top-level data from the JSON (e.g., metadata, task counts).
    """
    if os.path.isdir(file_path):
        json_files = glob.glob(os.path.join(file_path, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in the folder: {file_path}")
        latest_file = sorted(json_files)[-1]
        file_path = latest_file
        print(f"Opening the latest JSON file: {file_path}")
    else:
        file_path = file_path
    with open(file_path, 'r') as f:
        data = json.load(f)

    task_data = pd.DataFrame(data.get('task_statistics',))  # this is a large dict, one entry per task

    task_data['start_ms'] = task_data['start_ns'] / 1_000_000
    task_data['duration_ms'] = (task_data['stop_ns'] - task_data['start_ns']) / 1_000_000
    return task_data, data["metadata"]


def drop_constant_columns(df):
    """
    Drops all columns in the DataFrame that contain only a single unique value.

    Parameters:
    - df: Input pandas DataFrame.

    Returns:
    - A tuple:
        (1) A new DataFrame with constant columns removed.
        (2) A list of (column_name, unique_value) pairs for the dropped columns.
    """
    constant_info = []

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 1:
            constant_info.append((col, unique_vals[0]))

    df_reduced = df.drop(columns=[col for col, _ in constant_info])

    return df_reduced, constant_info

def group_stats_runtime(df):
    grp_atts = ['backend', 'variant_name', 'index_id']
    measures = ["executor_runtime_ms", "worker_utilization", "plan_construction_runtime_ms"]
    
    return df.groupby(grp_atts)[measures].mean().sort_values("executor_runtime_ms")

def group_stats_data_volume(large_df):
    grp_atts = ["b", "d", "index_id"]
    measures = ["total_uncompressed_list_size_GB","total_list_storage_costs_GB", "total_read_volume_GB"]

    return large_df.groupby(grp_atts)[measures].mean().sort_index()


def plot_indexed_scatter_seaborn(df, grp1="x", grp2="hue", symb=None, yscale="log", ymin=None, ymax=None, path="./", exp_suffix="", show_errorbars=False, show_means=True):
    """
    Creates one plot per attribute for a dataframe with either 3-level or 4-level MultiIndex.
    Uses grp1 for x-axis, grp2 for hue, and optionally symb to assign per-point markers.
    Aggregates over the remaining level (e.g., query) for mean/std.
    """

    path = Path(path)
    assert path.exists() and path.is_dir(), f"{path} does not exist or is no folder!"

    if not isinstance(df.index, pd.MultiIndex) or len(df.index.levels) not in [3, 4]:
        raise ValueError("DataFrame must have a 3- or 4-level MultiIndex")

    df_flat = df.reset_index()
    value_columns = df.columns

    # Colors for hues (grp2)
    unique_hues = df_flat[grp2].unique()
    palette = sns.color_palette('Set2', n_colors=len(unique_hues))
    color_mapping = dict(zip(unique_hues, palette))

    # Marker mapping if symb is given
    if symb:
        unique_symbs = df_flat[symb].unique()
        marker_styles = ['o', 's', '^', 'v', 'D', 'X', 'P', '*', '+', 'x']
        marker_mapping = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(unique_symbs)}
    else:
        df_flat["_dummy_symb"] = "_"
        symb = "_dummy_symb"
        marker_mapping = {"_": 'o'}

    # Dodge hue values
    dodge_amount = 0.15
    hue_order = list(unique_hues)
    hue_offsets = {
        hue_val: (i - (len(hue_order) - 1) / 2) * dodge_amount
        for i, hue_val in enumerate(hue_order)
    }

    for col in value_columns:
        plt.figure(figsize=(10, 6))

        # Compute x-axis positions
        x_vals = df_flat[grp1].unique()
        x_pos_map = {val: i for i, val in enumerate(x_vals)}

        # Plot raw points manually with jitter and markers
        for (x_val, hue_val, symb_val), subdf in df_flat.groupby([grp1, grp2, symb]):
            x_base = x_pos_map[x_val] + hue_offsets[hue_val]
            jitter = np.random.uniform(-0.05, 0.05, size=len(subdf))
            plt.scatter(
                x=np.full(len(subdf), x_base) + jitter,
                y=subdf[col],
                color=color_mapping[hue_val],
                marker=marker_mapping[symb_val],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8,
                label=hue_val
            )

        if show_means:
            # Group for mean/std
            grouped = df_flat.groupby([grp1, grp2, symb])[col]
            means = grouped.mean().reset_index()
            stds = grouped.std().reset_index()
            means = pd.merge(means, stds, on=[grp1, grp2, symb], suffixes=('', '_std'))

            for _, row in means.iterrows():
                x_val = row[grp1]
                hue_val = row[grp2]
                symb_val = row[symb]
                mean_val = row[col]
                std_val = row[f"{col}_std"]

                x_base = x_pos_map[x_val] + hue_offsets[hue_val]
                marker = marker_mapping[symb_val]
                color = color_mapping[hue_val]

                plt.scatter(
                    x=x_base,
                    y=mean_val,
                    color="grey",
                    edgecolor='black',
                    marker=marker,
                    s=50,
                    zorder=5
                )
                if show_errorbars:
                    if yscale == "log":
                        print(f"Warning: yscale is set to log, but error bars are shown for {col}!")
                    plt.errorbar(
                        x=x_base,
                        y=mean_val,
                        yerr=std_val,
                        fmt='none',
                        ecolor=color,
                        elinewidth=1.5,
                        capsize=4,
                        zorder=4
                    )

                y_offset = 0.3 * mean_val
                plt.text(
                    x=x_base,
                    y=mean_val + y_offset,
                    s=f"{mean_val:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color="black",
                    fontweight='bold',
                    zorder=6
                )

        # Final plot setup
        ax = plt.gca()
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=45, ha='right')

        # plt.title(f"{exp_suffix}: {col} over Aggregated Trials")
        plt.ylabel(col.replace("_", " "))
        plt.xlabel(grp1.replace("_", " "))

        plt.yscale(yscale)
        if ymin is not None or ymax is not None:
            plt.ylim(bottom=ymin if ymin is not None else None, top=ymax if ymax is not None else None)

        # Create legend for colors (hue)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend1 = plt.legend(by_label.values(), by_label.keys(), title=grp2, loc='upper right', frameon=False, facecolor='white')
        plt.gca().add_artist(legend1)

        # Create legend for marker styles (symb)
        import matplotlib.lines as mlines
        symb_handles = [
            mlines.Line2D([], [], color='black', marker=marker_mapping[val], linestyle='None',
                          markersize=8, label=str(val))
            for val in marker_mapping
        ]
        plt.legend(handles=symb_handles, title=symb if symb != "_dummy_symb" else None, loc='lower right', frameon=False, facecolor='white')
        plt.tight_layout()

        plt.savefig(path / f"{col}_{exp_suffix}.pdf")
        plt.close()




def plot_task_stats_aggregated(task_data, metadata, output_path=None, experiment_name=None):
    """
    Visualizes aggregated parallelism utilization using a horizontal stacked bar plot
    showing the total time spent by each worker on different task types and saves it to a PDF.

    Args:
        task_data (pd.DataFrame): DataFrame containing task statistics.
        metadata (dict): Dictionary containing metadata.
        output_path (str, optional): The location to write the plot pdf to. May be a file, then we place the new file next to it in the same folder.
    """
    if task_data.empty:
        print("No task data to visualize.")
        return
    if output_path is not None:
        output_path = Path(output_path)
        if experiment_name is not None:
            if output_path.is_file():
                output_path = output_path.parent
            if "execution_start" in metadata:
                time_str = "_"+metadata["execution_start"]
            else:
                time_str = ""
            output_pdf = output_path.joinpath(experiment_name+time_str+".pdf")
        else:
            assert output_path.exists(), output_path
            if output_path.is_file():
                output_pdf = output_path.stem+".pdf"
            else:
                output_pdf = output_path.joinpath("task_stats.pdf")

    # Convert relative times to milliseconds for easier aggregation
    task_data['start_ms'] = task_data['start_ns'] / 1_000_000
    task_data['duration_ms'] = (task_data['stop_ns'] - task_data['start_ns']) / 1_000_000

    # Count the number of tasks for each worker and type
    task_counts = task_data.groupby(['worker_id', 'type']).size().unstack(fill_value=0)

    # Group by worker ID and task type and calculate total duration
    aggregated_data = task_data.groupby(['worker_id', 'type'])['duration_ms'].sum().unstack(fill_value=pd.NA)

    if output_path is not None:
        # Identify where the task count was 0 and set the duration to NA
        for worker_id in aggregated_data.index:
            for task_type in aggregated_data.columns:
                if task_counts.at[worker_id, task_type] == 0:
                    aggregated_data.at[worker_id, task_type] = pd.NA
        task_data['task_duration_ns'] = task_data['stop_ns'] - task_data['start_ns']
        total_task_time_ns = task_data['task_duration_ns'].sum()
        num_workers = task_data['worker_id'].nunique()
        total_runtime_ns = metadata.get('execution_time_ns', 0)
        theoretical_available_time_ns = num_workers * total_runtime_ns
        worker_utilization = total_task_time_ns / theoretical_available_time_ns

        worker_ids = sorted(aggregated_data.index.tolist())
        task_types = aggregated_data.columns.tolist()        

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

        left = [0] * len(worker_ids) # Changed from bottom to left

        for task_type in task_types:
            durations = aggregated_data[task_type].fillna(0).tolist() # Fill NA with 0 for plotting
            ax.barh(worker_ids, durations, left=left, label=task_type) # Changed ax.bar to ax.barh, and bottom to left
            left = [l + d for l, d in zip(left, durations)] # Updated variable name

        ax.set_ylabel("Worker ID") # Changed from xlabel to ylabel
        ax.set_xlabel("Time (milliseconds)") # Changed from ylabel to xlabel
        title_str = "Aggregated Task Times"
        if experiment_name is not None:
            title_str += f"\nExperiment: \'{experiment_name}\'"
        title_str += f"\nWorker Utilization: {100*worker_utilization:.1f}%"

        ax.set_title(title_str)

        total_runtime_ns = metadata.get('execution_time_ns', 0)
        total_runtime_ms = total_runtime_ns / 1_000_000

        if total_runtime_ms > 0:
            ax.set_xlim(0, total_runtime_ms * 1.1) # Changed ylim to xlim
            ax.vlines(total_runtime_ms, worker_ids[0]-0.4, worker_ids[-1]+0.4, color='black', linestyle='--', linewidth=1.8, label='Total Runtime') # Changed hlines to vlines, and adjusted coordinates
            ax.set_yticks(worker_ids) # Changed xticks to yticks
        ax.legend(loc="lower left") # Adjusted legend location

        plt.tight_layout()

        # Save the plot to a PDF file
    
        plt.savefig(output_pdf)
        
        print(f"Aggregated task time plot saved to: {output_pdf}")
    return aggregated_data


def plot_worker_tasks(task_data, figure_path, type_color_mapping=None):
    """
    Plots a Gantt-style chart for worker tasks with task types determining the color.

    The DataFrame `task_data` must include:
      - 'worker_id': an identifier for each worker.
      - Either columns 'start_ms' and 'duration_ms' or columns 'start_ns' and 'stop_ns'
        (which are then used to compute start_ms and duration_ms).
      - 'type': a string representation of the task type (e.g., "Task::Leaf", "Task::DistributedIntersection", etc.)

    Parameters:
      task_data: pandas.DataFrame containing the task information.
      type_color_mapping: Optional dictionary mapping task type strings to colors.
                          If None, a default mapping is used.
    """
    
    # Define a default color mapping if one is not provided.
    if type_color_mapping is None:
        type_color_mapping = {
            "Task::Leaf": "lightblue",
            "Task::LeafUnionPrepare": "orange",
            "Task::LeafUnion": "green",
            "Task::ExpandedInit": "red",
            "Task::ExpandedIntersection": "purple",
            "Task::ExpandedSubtraction": "brown",
            "Task::DistributedIntersection": "pink",
            "Task::DistributedSubtraction": "grey",
            "Task::BigInnerUnion": "yellow",
            "Task::BigOuterUnion": "cyan",
            "Task::BigInnerIntersection": "magenta",
            "Task::BigOuterIntersection": "olive",
            "Task::TeamUnion": "teal",
        }
        
    # Compute 'start_ms' and 'duration_ms' if not already available.
    if 'start_ms' not in task_data.columns or 'duration_ms' not in task_data.columns:
        if 'start_ns' in task_data.columns and 'stop_ns' in task_data.columns:
            task_data = task_data.copy()
            task_data['start_ms'] = task_data['start_ns'] / 1e6
            task_data['duration_ms'] = (task_data['stop_ns'] - task_data['start_ns']) / 1e6
        else:
            raise ValueError("DataFrame must include either start_ms and duration_ms, or start_ns and stop_ns columns.")
    
    # Map worker IDs to y-axis positions.
    worker_ids = sorted(task_data['worker_id'].unique())
    worker_to_y = {worker: i for i, worker in enumerate(worker_ids)}
    
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw each task as a rectangle.
    for idx, row in task_data.iterrows():
        start = row['start_ms']
        duration = row['duration_ms']
        worker = row['worker_id']
        y = worker_to_y[worker]
        
        # Determine the color based on the task type.
        # If no type column is present or if the value is not in the mapping,
        # a default color is used.
        task_type = row.get("type", None)
        task_color = type_color_mapping.get(task_type, "skyblue")
        
        rect = patches.Rectangle((start, y - 0.4), duration, 0.8,
                                 facecolor=task_color, edgecolor="black", alpha=0.8)
        ax.add_patch(rect)
    
    # Set labels and title.
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Worker ID")
    ax.set_title("Task Utilization by Worker (Colored by Task Type)")
    
    # Configure y-axis ticks.
    ax.set_yticks(list(worker_to_y.values()))
    ax.set_yticklabels([str(worker) for worker in worker_ids])
    
    # Optionally add a legend if the 'task_type' column exists.
    if "type" in task_data.columns:
        unique_task_types = task_data['type'].unique()
        legend_patches = [patches.Patch(color=type_color_mapping.get(t, "skyblue"), label=t)
                          for t in unique_task_types]
        ax.legend(handles=legend_patches, title="Task Type")
    
    # Add grid lines for better readability.
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Set x and y limits to ensure all tasks are visible.
    min_time = task_data['start_ms'].min()
    max_time = (task_data['start_ms'] + task_data['duration_ms']).max()
    ax.set_xlim(min_time - 10, max_time + 10)
    ax.set_ylim(-1, len(worker_ids))
    
    fig.savefig(figure_path)








