#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import optuna
import time
import re
from dataclasses import dataclass, asdict
from typing import Optional

from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend
from O2GPU_autotuner.step_tuner import StepTuner
from O2GPU_autotuner.orchestrator import Orchestrator

@dataclass
class TunerConfig:
    output: str
    dataset: str
    nEvents: Optional[int]
    trials: Optional[int]
    startup: Optional[int]
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
    backend.rtc(rtc_cache=False)
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
    parser.add_argument("--time-budget", default="30m", help="Time budget for tuning: minutes (30m), hours (1h), or hh:mm (1:30)")
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
    if args.trials is not None:
        n_trials = args.trials
    else:
        n_trials, startup = estimate_iterations(backend, time_budget_sec)
    if args.startup is not None:
        if args.startup >= n_trials:
            print(f"[WARNING] Startup iterations ({args.startup}) >= total trials ({n_trials}), adjusting startup to {n_trials // 5}")
            startup = n_trials // 5
        startup = args.startup
    else:
        startup = max(1, int(0.15 * n_trials))
    print(f"[INFO] Running {n_trials} trials with {startup} startup trials.")
    
    step_tuners = {s: StepTuner(s, spaces[s], backend, output_dir, startup) for s in steps}
    orchestrator = Orchestrator(output_dir, step_tuners, backend, TUNER_PARAMETER_FILE)

    for iteration in range(n_trials):
        print(f"\n========== iteration {iteration} / {n_trials} ==========")
        
        orchestrator.ask()
        orchestrator.sample_params()

        ok = orchestrator.rt_compile_for_HW_stats()
        if not ok:
            for sTuner in step_tuners.values():
                sTuner.bad_iteration = True

        ok = orchestrator.compute_full_bounds()
        if not ok:
            for sTuner in step_tuners.values():
                sTuner.bad_iteration = True
        
        success = orchestrator.compile_and_run_with_full_bounds()
        if not success:
            for sTuner in step_tuners.values():
                sTuner.bad_iteration = True
        
        timings = orchestrator.timings
    
        for sTuner in step_tuners.values():
            value = timings.get(sTuner.name, float("inf"))
            sTuner.tell(value)
        
        orchestrator.reset_bad_iterations()