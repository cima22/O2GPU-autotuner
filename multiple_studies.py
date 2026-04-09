#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import optuna

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
# PARAM BUILDING (your logic)
# =========================

def build_kernel_params(trial, tune_config, backend, kernel_name):
    kernels_param_space = {}

    for param_name, spec in tune_config.items():
        full_name = f"{kernel_name}_{param_name}"

        if param_name.startswith("PAR_"):
            if spec["type"] == "range":
                kernels_param_space[param_name] = trial.suggest_int(
                    full_name, spec["min"], spec["max"]
                )
            elif spec["type"] == "values":
                kernels_param_space[param_name] = trial.suggest_categorical(
                    full_name, spec["values"]
                )
        else:
            blocks_per_sm = None
            block_value = None

            if "blocks_per_sm" in spec:
                s = spec["blocks_per_sm"]
                name = f"{full_name}_blocks_per_sm"

                if s["type"] == "range":
                    blocks_per_sm = trial.suggest_int(
                        name, s["min"], s["max"], step=s.get("step", 1)
                    )
                elif s["type"] == "values":
                    blocks_per_sm = trial.suggest_categorical(
                        name, s["values"]
                    )

            if "block_size" in spec:
                s = spec["block_size"]
                name = f"{full_name}_block_size"

                if s["type"] == "range":
                    block_value = trial.suggest_int(
                        name,
                        s["min"] * backend.warpSize,
                        s["max_value"],
                        step=s.get("step", 1) * backend.warpSize
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
            if (
                spec["blocks_per_sm"] * spec["block_size"]
                > backend.maxThreadsPerMultiProcessor
                and kernel_name != "tracklet"
            ):
                return True
    return False


# =========================
# BACKEND (YOU IMPLEMENT)
# =========================

def run_backend_once(all_kernel_params, backend):
    """
    all_kernel_params:
    {
        "kernelA": {...},
        "kernelB": {...}
    }

    You must:
    - write params to config/header
    - run application ONCE
    - extract per-kernel timings

    Return:
    {
        "kernelA": float,
        "kernelB": float,
    }
    """

    # TODO: implement real execution
    print("\n[DEBUG] Running backend once with:")
    for k, v in all_kernel_params.items():
        print(f"{k}: {v}")
    merged = flatten_params(all_kernel_params)
    try:
        backend.update_param_file(merged, TUNER_PARAMETER_FILE)
        backend.profile_benchmark(TUNER_DATASET)
        success = True
    except Exception as e:
        print(f"[ERROR] Backend failed: {e}")
        success = False
    return success

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_DIR_ENV)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--startup", type=int, required=True)
    args = parser.parse_args()

    output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(TUNER_WORKDIR)

    backend = BenchmarkBackend(output_dir)

    # Discover + load
    kernels = discover_kernels(TUNE_SPACE_DIR)
    spaces = load_spaces(TUNE_SPACE_DIR, kernels)

    print("Discovered kernels:")
    for k in kernels:
        print(f"  - {k}")

    # Create studies
    studies = {
        k: optuna.create_study(
            study_name=k,
            direction="minimize",
            sampler=make_sampler(args.startup),
            storage=f"sqlite:///{output_dir}/{k}.db",
            load_if_exists=True,
        )
        for k in kernels
    }

    # =========================
    # OPT LOOP
    # =========================

    for step in range(args.trials):
        print(f"\n========== STEP {step} ==========")

        trials = {k: studies[k].ask() for k in kernels}
        all_params = {}
        valid = {}

        # Build params
        for k in kernels:
            params = build_kernel_params(
                trials[k],
                spaces[k],
                backend,
                k
            )

            if is_invalid_config(params, backend, k):
                valid[k] = False
            else:
                valid[k] = True

            all_params[k] = params

        # 🚀 RUN ONCE
        outcome = run_backend_once(all_params, backend)

        # Tell Optuna
        for k in kernels:
            if not valid[k]:
                studies[k].tell(trials[k], float("inf"))
            else:
                mean, _ = backend.compute_step_mean_time(k, all_params[k], TUNER_DATASET)
                studies[k].tell(trials[k], mean)
                print(f"{k}: {mean:.6f}")

    print("\n========== DONE ==========")

    # Print best
    for k in kernels:
        print(f"\nBest for {k}:")
        print(studies[k].best_trial)

    os.chdir(original_cwd)


if __name__ == "__main__":
    main()
