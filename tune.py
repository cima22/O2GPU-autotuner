#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import optuna
import time
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend

TUNE_SPACE_DIR = os.getenv("TUNE_SPACE_DIR", os.path.join(os.path.dirname(__file__), "tune_spaces"))
TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../standalone"))
TUNER_DATASET = os.getenv("TUNER_DATASET", "o2-pbpb-47kHz-32")
TUNER_PARAMETER_FILE = os.getenv("TUNER_PARAMETER_FILE", os.path.join(TUNER_WORKDIR, "defaultParams.h"))
OUTPUT_DIR_ENV = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "tuning_results"))

def discover_steps(tune_space_dir):
    steps = []
    for f in os.listdir(tune_space_dir):
        if f.endswith(".yaml") and not f.startswith("_"):
            steps.append(os.path.splitext(f)[0])
    steps.sort()
    return steps

def load_spaces(tune_space_dir, steps):
    spaces = {}
    for s in steps:
        path = os.path.join(tune_space_dir, f"{s}.yaml")
        with open(path, "r") as f:
            spaces[s] = yaml.safe_load(f)
    return spaces

def flatten_params(all_kernel_params):
    merged = {}
    for params in all_kernel_params.values():
        merged.update(params)
    return merged

def make_sampler(startup):
    return optuna.samplers.TPESampler(n_startup_trials=startup, constant_liar=False, multivariate=True)

def build_step_params(trial, tune_config, backend):
    kernels_param_space = {}
    for param_name, spec in tune_config.items():
        full_name = param_name
        if param_name.startswith("PAR_"):
            if spec["type"] == "range":
                kernels_param_space[param_name] = trial.suggest_int(full_name, spec["min"], spec["max"])
            elif spec["type"] == "values":
                kernels_param_space[param_name] = trial.suggest_categorical(full_name, spec["values"])
        else:
            blocks_per_sm = None
            block_value = None
            if "blocks_per_sm" in spec:
                s = spec["blocks_per_sm"]
                name = f"{full_name}_blocks_per_sm"
                if s["type"] == "range":
                    blocks_per_sm = trial.suggest_int(name, s["min"], s["max"], step=s.get("step", 1))
                elif s["type"] == "values":
                    blocks_per_sm = trial.suggest_categorical(name, s["values"])
            if "block_size" in spec:
                s = spec["block_size"]
                name = f"{full_name}_block_size"
                if s["type"] == "range":
                    block_value = trial.suggest_int(name, s["min"] * backend.warpSize, s["max_value"], step=s.get("step", 1) * backend.warpSize)
                elif s["type"] == "values":
                    warp_values = [v for v in s["values"] if v % backend.warpSize == 0]
                    block_value = trial.suggest_categorical(name, warp_values)
            kernels_param_space[param_name] = {}
            if blocks_per_sm is not None:
                kernels_param_space[param_name]["blocks_per_sm"] = blocks_per_sm
            if block_value is not None:
                kernels_param_space[param_name]["block_size"] = block_value
    return kernels_param_space

def is_invalid_config(kernels_param_space, backend, kernel_name):
    for param_name, spec in kernels_param_space.items():
        if isinstance(spec, dict) and "blocks_per_sm" in spec and "block_size" in spec:
            if spec["blocks_per_sm"] * spec["block_size"] > backend.maxThreadsPerMultiProcessor and kernel_name != "tracklet":
                return True
    return False

def run_backend(all_kernel_params, backend, iteration, output_dir, steps):
    print("\n[DEBUG] Running backend with:")
    for k, v in all_kernel_params.items():
        print(f"{k}: {v}")

    merged = flatten_params(all_kernel_params)
    run_log = os.path.join(output_dir, f"run_{iteration}.log")
    timings = {}

    try:
        try:
            backend.update_param_file(merged, TUNER_PARAMETER_FILE, log_file=run_log)
            backend.profile_benchmark(TUNER_DATASET, run_log_file=run_log)
            success = True
        except (RuntimeError, TimeoutError) as e:
            print(f"[INFO] Backend failed: {e}")
            success = False

        if success:
            for s in steps:
                mean, _ = backend.compute_step_mean_time(s, all_kernel_params[s], TUNER_DATASET)
                timings[s] = mean
            return timings

        print("[INFO] Parsing log for failing kernels")
        bad_steps = backend.detectFailingKernels(run_log, steps)
        if not bad_steps:
            print("[WARNING] No kernel identified, mark all as bad")
            return {s: float("inf") for s in steps}
        print(f"[INFO] Detected failing kernels: {bad_steps}")

        for s in bad_steps:
            timings[s] = float("inf")
        good_steps = [s for s in steps if s not in bad_steps]

        if good_steps:
            print(f"[INFO] Retrying without bad kernels: {good_steps}")
            subset = {s: all_kernel_params[s] for s in good_steps}
            try:
                backend.update_param_file(flatten_params(subset), TUNER_PARAMETER_FILE, log_file=run_log)
                backend.profile_benchmark(TUNER_DATASET, run_log_file=run_log)
                for s in good_steps:
                    mean, _ = backend.compute_step_mean_time(s, all_kernel_params[s], TUNER_DATASET)
                    timings[s] = mean
            except (RuntimeError, TimeoutError) as e:
                print(f"[WARNING] Retry failed: {e}")
                for s in good_steps:
                    timings[s] = float("inf")
        return timings
    finally:
        try:
            if os.path.exists(run_log):
                os.remove(run_log)
        except Exception as e:
            print(f"[WARNING] Could not remove run log: {e}")

def parse_time_budget(time_str):
    time_str = time_str.strip().lower()
    if re.match(r"^\d+$", time_str):
        return int(time_str) * 60
    elif time_str.endswith("m"):
        return int(time_str[:-1]) * 60
    elif time_str.endswith("h"):
        return int(time_str[:-1]) * 3600
    elif re.match(r"^\d+:\d+$", time_str):
        h, m = map(int, time_str.split(":"))
        return h * 3600 + m * 60
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def estimate_iterations(backend, time_budget_sec):
    print("[INFO] Running empty iteration to estimate timing...")
    t0 = time.time()
    backend.update_param_file({}, TUNER_PARAMETER_FILE, log_file="/tmp/empty_run.log")
    backend.profile_benchmark(TUNER_DATASET)
    t1 = time.time()
    iter_time = t1 - t0
    print(f"[INFO] Estimated iteration time: {iter_time:.2f}s")
    n_trials = max(1, int(time_budget_sec / iter_time))
    n_startup = max(1, int(0.15 * n_trials))
    print(f"[INFO] Total trials: {n_trials}, startup trials: {n_startup}")
    return n_trials, n_startup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_DIR_ENV)
    parser.add_argument("--trials", type=int, help="Override automatically computed number of trials")
    parser.add_argument("--startup", type=int, help="Override automatically computed number of startup iterations")
    parser.add_argument("--time_budget", default="30m", help="Time budget for tuning: minutes (30m), hours (1h), or hh:mm (1:30)")
    args = parser.parse_args()

    output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(TUNER_WORKDIR)

    backend = BenchmarkBackend(output_dir)
    steps = discover_steps(TUNE_SPACE_DIR)
    spaces = load_spaces(TUNE_SPACE_DIR, steps)

    print("Discovered steps:")
    for s in steps:
        print(f"  - {s}")

    time_budget_sec = parse_time_budget(args.time_budget)
    trials, startup = estimate_iterations(backend, time_budget_sec)
    if args.trials is not None:
        trials = args.trials
    if args.startup is not None:
        startup = args.startup
    print(f"[INFO] Running {trials} trials with {startup} startup trials.")

    studies = {s: optuna.create_study(study_name=s, direction="minimize", sampler=make_sampler(startup), storage=f"sqlite:///{output_dir}/{s}.db", load_if_exists=True) for s in steps}

    for iteration in range(trials):
        print(f"\n========== iteration {iteration} ==========")
        trials = {s: studies[s].ask() for s in steps}
        all_params = {}
        valid = {}

        for s in steps:
            params = build_step_params(trials[s], spaces[s], backend)
            valid[s] = not is_invalid_config(params, backend, s)
            all_params[s] = params

        timings = run_backend(all_params, backend, iteration, output_dir, steps)

        for s in steps:
            if not valid[s]:
                studies[s].tell(trials[s], float("inf"))
                print(f"{s}: invalid configuration → inf")
            else:
                value = timings.get(s, float("inf"))
                studies[s].tell(trials[s], value)
                print(f"{s}: {value:.6f}")

    print("\n========== DONE ==========")
    for s in steps:
        print(f"\nBest for {s}:")
        print(studies[s].best_trial)

    os.chdir(original_cwd)

if __name__ == "__main__":
    main()
