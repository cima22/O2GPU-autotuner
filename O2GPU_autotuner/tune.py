#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import optuna
import time
import re
from dataclasses import dataclass, asdict

from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend

@dataclass
class TunerConfig:
    output: str
    dataset: str
    nEvents: int | None
    trials: int | None
    startup: int | None
    time_budget: str
    parameter_file: str

TUNE_SPACE_DIR = os.getenv("TUNE_SPACE_DIR", os.path.join(os.path.dirname(__file__), "tune_spaces"))
TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../../standalone"))
TUNER_DATASET = os.getenv("TUNER_DATASET", "47kHz")
TUNER_PARAMETER_FILE = os.getenv("TUNER_PARAMETER_FILE", os.path.join(os.path.dirname(__file__), "defaults", "defaultParamsNVIDIA.h"))
OUTPUT_DIR_ENV = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "tuning_results"))

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
        if param_name.startswith("PAR_"):
            if spec["type"] == "range":
                kernels_param_space[param_name] = trial.suggest_int(param_name, spec["min"], spec["max"])
            elif spec["type"] == "values":
                kernels_param_space[param_name] = trial.suggest_categorical(param_name, spec["values"])
            continue

        block_size = None
        blocks_per_sm = None
        bs_spec = spec.get("block_size", None)
        if bs_spec is not None:
            if bs_spec["type"] == "single":
                block_size = bs_spec["values"][backend.backend]
            elif bs_spec["type"] == "range":
                name = f"{param_name}_block_size"
                block_size = trial.suggest_int(name, bs_spec["min"] * backend.warpSize, bs_spec["max_value"], step=bs_spec.get("step", 1) * backend.warpSize)
            elif bs_spec["type"] == "values":
                name = f"{param_name}_block_size"
                warp_values = [v for v in bs_spec["values"] if v % backend.warpSize == 0]
                block_size = trial.suggest_categorical(name, warp_values)
            block_size = (block_size // backend.warpSize) * backend.warpSize

        if block_size is None:
            raise ValueError(f"block_size not defined for {param_name}")

        stats = backend.get_kernel_HW_stats(param_name)
        shm = stats["shared_memory"]
        max_bpsm_threads = backend.GPUlimits["max_threads_per_sm"] // block_size
        max_bpsm_hw     = backend.GPUlimits["max_blocks_per_sm"]
        max_bpsm_shm    = backend.GPUlimits["shared_mem_per_sm"] // shm if shm > 0 else max_bpsm_hw
        max_bpsm        = min(max_bpsm_threads, max_bpsm_hw, max_bpsm_shm)

        if max_bpsm < 1:
            raise optuna.TrialPruned()

        name = f"{param_name}_blocks_per_sm"
        blocks_per_sm = trial.suggest_int(name, 1, max_bpsm, step=1)
        kernels_param_space[param_name] = {"block_size": block_size, "blocks_per_sm": blocks_per_sm}

    return kernels_param_space

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
            backend.profile_benchmark(run_log_file=run_log)
            success = True
        except (RuntimeError, TimeoutError) as e:
            print(f"[INFO] Backend failed: {e}")
            success = False

        if success:
            for s in steps:
                mean, _ = backend.compute_step_mean_time(s, all_kernel_params[s])
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
                backend.profile_benchmark(run_log_file=run_log)
                for s in good_steps:
                    mean, _ = backend.compute_step_mean_time(s, all_kernel_params[s])
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
    backend.profile_benchmark()
    t1 = time.time()
    iter_time = t1 - t0
    print(f"[INFO] Estimated iteration time: {iter_time:.2f}s")
    n_trials = max(1, int(time_budget_sec / iter_time))
    n_startup = max(1, int(0.15 * n_trials))
    print(f"[INFO] Total trials: {n_trials}, startup trials: {n_startup}")
    return n_trials, n_startup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_DIR_ENV, help="Directory to store tuning results and logs")
    parser.add_argument("--dataset", default=TUNER_DATASET, help="Dataset to use for benchmarking")
    parser.add_argument("--nEvents", type=int, help="Override number of events for benchmarking")
    parser.add_argument("--trials", type=int, help="Override automatically computed number of trials")
    parser.add_argument("--startup", type=int, help="Override automatically computed number of startup iterations")
    parser.add_argument("--time_budget", default="30m", help="Time budget for tuning: minutes (30m), hours (1h), or hh:mm (1:30)")
    args = parser.parse_args()

    output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    original_cwd = os.getcwd()

    config = TunerConfig(
        output=os.path.realpath(args.output),
        dataset=str(args.dataset),
        nEvents=args.nEvents,
        trials=args.trials,
        startup=args.startup,
        time_budget=args.time_budget,
        parameter_file=TUNER_PARAMETER_FILE,
    )
    config_path = os.path.join(output_dir, "run_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)

    os.chdir(TUNER_WORKDIR)
    backend = BenchmarkBackend(output_dir)
    dataset = str(args.dataset)
    backend.dataset = dataset
    if args.nEvents is not None:
        backend.num_events = args.nEvents
    steps = discover_steps(TUNE_SPACE_DIR)
    spaces = load_spaces(TUNE_SPACE_DIR, steps)

    print("Discovered steps:")
    for s in steps:
        print(f"  - {s}")

    time_budget_sec = parse_time_budget(args.time_budget)
    n_trials, startup = estimate_iterations(backend, time_budget_sec)
    if args.trials is not None:
        n_trials = args.trials
    if args.startup is not None:
        startup = args.startup
    print(f"[INFO] Running {n_trials} trials with {startup} startup trials.")

    studies = {s: optuna.create_study(study_name=s, direction="minimize", sampler=make_sampler(startup), storage=f"sqlite:///{output_dir}/{s}.db", load_if_exists=True) for s in steps}

    for iteration in range(n_trials):
        print(f"\n========== iteration {iteration} / {n_trials} ==========")
        trials = {s: studies[s].ask() for s in steps}
        all_params = {}

        for s in steps:
            params = build_step_params(trials[s], spaces[s], backend)
            all_params[s] = params
        timings = run_backend(all_params, backend, iteration, output_dir, steps)

        for s in steps:
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
