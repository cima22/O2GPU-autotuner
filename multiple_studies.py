#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import optuna
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend

# =========================
# ENV CONFIG
# =========================
TUNE_SPACE_DIR = os.getenv("TUNE_SPACE_DIR", os.path.join(os.path.dirname(__file__), "tune_spaces"))
TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../standalone"))
TUNER_DATASET = os.getenv("TUNER_DATASET", "o2-pbpb-47kHz-32")
TUNER_PARAMETER_FILE = os.getenv("TUNER_PARAMETER_FILE", os.path.join(TUNER_WORKDIR, "defaultParams.h"))
OUTPUT_DIR_ENV = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "tuning_results"))

# =========================
# HELPERS
# =========================
def discover_kernels(tune_space_dir):
    kernels = []
    for f in os.listdir(tune_space_dir):
        if f.endswith(".yaml") and not f.startswith("_"):
            kernels.append(os.path.splitext(f)[0])
    kernels.sort()
    return kernels

def load_spaces(tune_space_dir, kernels):
    spaces = {}
    for k in kernels:
        path = os.path.join(tune_space_dir, f"{k}.yaml")
        with open(path, "r") as f:
            spaces[k] = yaml.safe_load(f)
    return spaces

def flatten_params(all_kernel_params):
    merged = {}
    for params in all_kernel_params.values():
        merged.update(params)
    return merged

def make_sampler(startup):
    return optuna.samplers.TPESampler(
        n_startup_trials=startup,
        constant_liar=False,
        multivariate=True
    )

# =========================
# PARAM BUILDING
# =========================
def build_kernel_params(trial, tune_config, backend, kernel_name):
    kernels_param_space = {}
    for param_name, spec in tune_config.items():
        full_name = f"{kernel_name}_{param_name}"
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
                    block_value = trial.suggest_int(
                        name, s["min"] * backend.warpSize, s["max_value"], step=s.get("step", 1) * backend.warpSize
                    )
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

# =========================
# BACKEND EXECUTION
# =========================
def run_backend_once(all_kernel_params, backend, step, output_dir, kernels):
    """
    Execute backend once with all kernel params.
    Returns: dict {kernel_name: mean_time or inf}
    """
    print("\n[DEBUG] Running backend once with:")
    for k, v in all_kernel_params.items():
        print(f"{k}: {v}")

    merged = flatten_params(all_kernel_params)
    run_log = os.path.join(output_dir, f"run_{step}.log")
    timings = {}

    try:
        # Run benchmark
        try:
            backend.update_param_file(merged, TUNER_PARAMETER_FILE, log_file=run_log)
            backend.profile_benchmark(TUNER_DATASET, run_log_file=run_log)
            success = True
        except (RuntimeError, TimeoutError) as e:
            print(f"[INFO] Backend failed: {e}")
            success = False

        # SUCCESS
        if success:
            for k in kernels:
                mean, _ = backend.compute_step_mean_time(k, all_kernel_params[k], TUNER_DATASET)
                timings[k] = mean
            return timings

        # FAILURE → parse log
        print("[INFO] Parsing log for failing kernels")
        bad_kernels = backend.detectFailingKernels(run_log, kernels)
        if not bad_kernels:
            print("[WARNING] No kernel identified → mark all as bad")
            return {k: float("inf") for k in kernels}
        print(f"[INFO] Detected failing kernels: {bad_kernels}")

        for k in bad_kernels:
            timings[k] = float("inf")
        good_kernels = [k for k in kernels if k not in bad_kernels]

        # Retry good kernels
        if good_kernels:
            print(f"[INFO] Retrying without bad kernels: {good_kernels}")
            subset = {k: all_kernel_params[k] for k in good_kernels}
            try:
                backend.update_param_file(flatten_params(subset), TUNER_PARAMETER_FILE, log_file=run_log)
                backend.profile_benchmark(TUNER_DATASET, run_log_file=run_log)
                for k in good_kernels:
                    mean, _ = backend.compute_step_mean_time(k, all_kernel_params[k], TUNER_DATASET)
                    timings[k] = mean
            except (RuntimeError, TimeoutError) as e:
                print(f"[WARNING] Retry failed: {e}")
                for k in good_kernels:
                    timings[k] = float("inf")
        return timings
    finally:
        # Always clean the single-run log
        try:
            if os.path.exists(run_log):
                os.remove(run_log)
        except Exception as e:
            print(f"[WARNING] Could not remove run log: {e}")

# =========================
# TIME-BASED ITERATION CALC
# =========================

def parse_time_budget(time_str):
    """
    Parse a time string:
        - integer + m/h → minutes or hours
        - hh:mm → hours:minutes
    Returns: total seconds
    """
    import re

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
    """
    Do a single run to estimate mean time per iteration.
    Pass an empty config to backend.
    Returns: n_trials, n_startup_trials
    """
    print("[INFO] Running empty iteration to estimate timing...")
    import time

    t0 = time.time()
    backend.update_param_file({}, TUNER_PARAMETER_FILE)  # empty dictionary
    backend.profile_benchmark(TUNER_DATASET)
    t1 = time.time()

    iter_time = t1 - t0
    print(f"[INFO] Estimated iteration time: {iter_time:.2f}s")

    n_trials = max(1, int(time_budget_sec / iter_time))
    n_startup = max(1, int(0.15 * n_trials))
    print(f"[INFO] Total trials: {n_trials}, startup trials: {n_startup}")

    return n_trials, n_startup

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_DIR_ENV)
    #parser.add_argument("--trials", type=int, required=True)
    #parser.add_argument("--startup", type=int, required=True)
    parser.add_argument("--time_budget", default="30m", help="Time budget for tuning: minutes (30m), hours (1h), or hh:mm (1:30)")
    args = parser.parse_args()

    output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(TUNER_WORKDIR)

    backend = BenchmarkBackend(output_dir)
    kernels = discover_kernels(TUNE_SPACE_DIR)
    spaces = load_spaces(TUNE_SPACE_DIR, kernels)

    print("Discovered kernels:")
    for k in kernels:
        print(f"  - {k}")
    time_budget_sec = parse_time_budget(args.time_budget)
    trials, startup = estimate_iterations(backend, time_budget_sec)

    print(f"[INFO] Running {trials} trials with {startup} startup trials.")
    studies = {k: optuna.create_study(
        study_name=k,
        direction="minimize",
        sampler=make_sampler(startup),
        storage=f"sqlite:///{output_dir}/{k}.db",
        load_if_exists=True
    ) for k in kernels}

    # =========================
    # OPT LOOP
    # =========================
    for step in range(trials):
        print(f"\n========== STEP {step} ==========")
        trials = {k: studies[k].ask() for k in kernels}
        all_params = {}
        valid = {}

        # Build trial params
        for k in kernels:
            params = build_kernel_params(trials[k], spaces[k], backend, k)
            valid[k] = not is_invalid_config(params, backend, k)
            all_params[k] = params

        # Run backend ONCE
        timings = run_backend_once(all_params, backend, step, output_dir, kernels)

        # Tell Optuna
        for k in kernels:
            if not valid[k]:
                studies[k].tell(trials[k], float("inf"))
                print(f"{k}: invalid configuration → inf")
            else:
                value = timings.get(k, float("inf"))
                studies[k].tell(trials[k], value)
                print(f"{k}: {value:.6f}")

    # Print best results
    print("\n========== DONE ==========")
    for k in kernels:
        print(f"\nBest for {k}:")
        print(studies[k].best_trial)

    os.chdir(original_cwd)


if __name__ == "__main__":
    main()
