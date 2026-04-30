#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import optuna
from optuna.study import get_all_study_summaries

from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend

TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../../standalone"))

def load_config(output_dir):
    config_path = os.path.join(output_dir, "run_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found in {output_dir}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_best_from_db(db_path):
    storage = f"sqlite:///{db_path}"
    summaries = get_all_study_summaries(storage=storage)
    if not summaries:
        raise RuntimeError("No studies found in DB")

    study_name = summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial

    return study_name, best.value, best.params, best.user_attrs

def load_trial_from_db(db_path):
    storage = f"sqlite:///{db_path}"
    summaries = get_all_study_summaries(storage=storage)
    if not summaries:
        raise RuntimeError("No studies found in DB")

    study_name = summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage)

    trial = study.trials[0]  # 👈 direct access
    return study_name, trial.value, trial.params, trial.user_attrs

def reshape_config(flat_params, user_attrs):
    kernel_updates = {}
    macro_updates = {}
    # macros still from params
    for key, value in flat_params.items():
        if key.startswith("PAR_"):
            macro_updates[key] = value
    # everything else from user_attrs ONLY
    for key, value in user_attrs.items():
        if key.endswith("_block_size"):
            kernel = key[:-len("_block_size")]
            kernel_updates.setdefault(kernel, {})["block_size"] = value
        elif key.endswith("_blocks_per_sm"):
            kernel = key[:-len("_blocks_per_sm")]
            kernel_updates.setdefault(kernel, {})["blocks_per_sm"] = value
    # validation
    for k, v in kernel_updates.items():
        if "block_size" not in v or "blocks_per_sm" not in v:
            raise RuntimeError(f"Incomplete kernel config: {k}")

    return {**kernel_updates, **macro_updates}

def main():
    parser = argparse.ArgumentParser(description="Extract best configs from Optuna studies and write param and header files.")
    parser.add_argument("directory", help="Tuning directory")
    args = parser.parse_args()
    workdir = os.path.realpath(args.directory)
    if not os.path.isdir(workdir):
        print(f"[ERROR] Not a directory: {workdir}")
        return
    config = load_config(workdir)
    dataset = (config["dataset"])
    db_files = [f for f in os.listdir(workdir) if f.endswith(".db")]
    if not db_files:
        print("[ERROR] No .db files found")
        return
    print("\n========== EXTRACTING BEST CONFIGS ==========\n")
    merged_config = {}
    merged_user_attrs = {}
    for db in sorted(db_files):
        db_path = os.path.join(workdir, db)
        try:
            study_name, value, params, user_attrs = load_best_from_db(db_path)
            print(f"{db}:")
            print(f"  study: {study_name}")
            print(f"  best value: {value}")
            print(f"  params: {params}")
            print(f"  user attrs: {user_attrs}\n")
            for k, v in params.items():
                if k in merged_config:
                    print(f"[WARNING] Overwriting key: {k}")
                merged_config[k] = v
            for k, v in user_attrs.items():
                if k in merged_user_attrs:
                    print(f"[WARNING] Overwriting key: {k}")
                merged_user_attrs[k] = v
        except Exception as e:
            print(f"[ERROR] Failed reading {db}: {e}")

    print("\n========== WRITING PARAM FILE ==========\n")
    dump_path = os.path.join(workdir, "optimized.par")
    header_path = os.path.join(workdir, "optimized.h")
    original_cwd = os.getcwd()
    os.chdir(TUNER_WORKDIR)
    param_file = os.path.realpath(str(config["parameter_file"]))
    reshaped_config = reshape_config(merged_config, merged_user_attrs)
    try:
        backend = BenchmarkBackend(workdir)
        backend.dataset = dataset
        backend.num_events = config["nEvents"]
        print("[INFO] Running backend to verify performance...")
        backend.update_param_file({}, param_file, dump_path=dump_path)
        def_mean, def_std_dev = backend.get_sync_mean_time(dump=dump_path)
        backend.update_param_file(reshaped_config, param_file, modified_header_path = header_path, dump_path=dump_path)
        print(f"[INFO] Parameter file written to: {header_path}")
        print(f"[INFO] Dump written to: {dump_path}")
        opt_mean, opt_std_dev = backend.get_sync_mean_time(dump=dump_path)
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
