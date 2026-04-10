#!/usr/bin/env python3

import os
import sys


import argparse
import optuna
from optuna.study import get_all_study_summaries

from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend

TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../../standalone"))
TUNER_DATASET = os.getenv("TUNER_DATASET", "47kHz")
TUNER_PARAMETER_FILE = os.getenv("TUNER_PARAMETER_FILE", os.path.join(os.path.dirname(__file__), "defaults", "defaultParamsNVIDIA.h"))
OUTPUT_DIR_ENV = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "tuning_results"))

def load_best_from_db(db_path):
    storage = f"sqlite:///{db_path}"
    summaries = get_all_study_summaries(storage=storage)
    if not summaries:
        raise RuntimeError("No studies found in DB")
    study_name = summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial
    return study_name, best.value, best.params

def reshape_config(flat_config):
    kernel_updates = {}
    macro_updates = {}

    for key, value in flat_config.items():

        if key.startswith("PAR_"):
            macro_updates[key] = value
            continue

        if key.endswith("_block_size"):
            kernel = key.replace("_block_size", "")
            kernel_updates.setdefault(kernel, {})["block_size"] = value

        elif key.endswith("_blocks_per_sm"):
            kernel = key.replace("_blocks_per_sm", "")
            kernel_updates.setdefault(kernel, {})["blocks_per_sm"] = value

    return {**kernel_updates, **macro_updates}

def main():
    parser = argparse.ArgumentParser(description="Extract best configs from Optuna studies and write param and header files.")
    parser.add_argument("directory", help="Tuning directory")
    parser.add_argument("--param-file", default=TUNER_PARAMETER_FILE, help="Output header file")
    parser.add_argument("--dataset", default=TUNER_DATASET, help="Dataset to use for benchmarking")
    args = parser.parse_args()
    workdir = os.path.realpath(args.directory)
    dataset = args.dataset
    if not os.path.isdir(workdir):
        print(f"[ERROR] Not a directory: {workdir}")
        return
    db_files = [f for f in os.listdir(workdir) if f.endswith(".db")]
    if not db_files:
        print("[ERROR] No .db files found")
        return
    print("\n========== EXTRACTING BEST CONFIGS ==========\n")
    merged_config = {}
    for db in sorted(db_files):
        db_path = os.path.join(workdir, db)
        try:
            study_name, value, params = load_best_from_db(db_path)
            print(f"{db}:")
            print(f"  study: {study_name}")
            print(f"  best value: {value}")
            print(f"  params: {params}\n")
            for k, v in params.items():
                if k in merged_config:
                    print(f"[WARNING] Overwriting key: {k}")
                merged_config[k] = v
        except Exception as e:
            print(f"[ERROR] Failed reading {db}: {e}")

    print("\n========== WRITING PARAM FILE ==========\n")
    dump_path = os.path.join(workdir, "optimized.par")
    original_cwd = os.getcwd()
    os.chdir(TUNER_WORKDIR)
    param_file = os.path.realpath(args.param_file)
    reshaped_config = reshape_config(merged_config)
    try:
        backend = BenchmarkBackend(workdir)
        backend.update_param_file(reshaped_config, param_file, dump_path=dump_path)
        print(f"[INFO] Parameter file written to: {param_file}")
        print(f"[INFO] Dump written to: {dump_path}")
        print("[INFO] Running backend to verify performance...")

        def_mean, def_std_dev = backend.get_sync_mean_time(dump=None, dataset=dataset)
        opt_mean, opt_std_dev = backend.get_sync_mean_time(dump=dump_path, dataset=dataset)
        print("\n========== TIMING RESULTS ==========")
        print(f"[DEFAULT] mean = {def_mean:.6f} s | std = {def_std_dev:.6f} s")
        print(f"[OPTIMIZED] mean = {opt_mean:.6f} s | std = {opt_std_dev:.6f} s")

        if opt_mean > def_mean:
            slowdown = opt_mean - def_mean
            slowdown_pct = 100.0 * slowdown / def_mean
            raise RuntimeError(f"\nOptimization FAILED: slower configuration detected\nDefault mean : {def_mean:.6f} s\nOptimized mean: {opt_mean:.6f} s\nSlowdown     : +{slowdown:.6f} s ({slowdown_pct:.2f}%)")

        gain = def_mean - opt_mean
        speedup = def_mean / opt_mean if opt_mean > 0 else float("inf")
        improvement_pct = 100.0 * gain / def_mean
        print("\nOptimization SUCCESS")
        print(f"Gain      = {gain:.6f} s")
        print(f"Speedup   = {speedup:.3f}x")
        print(f"Improvement = {improvement_pct:.2f}%")
        print("====================================\n")

    except Exception as e:
        print(f"[ERROR] Failed to test optimised configuration: {e}")

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
