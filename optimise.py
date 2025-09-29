import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TUNE_SPACE_PATH = os.getenv("TUNE_SPACE_PATH", os.path.join(os.path.dirname(__file__), "tune_space.yaml"))
TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../standalone"))

with open(TUNE_SPACE_PATH, "r") as f:
    tune_config = yaml.safe_load(f)

from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend


def optimise(trial):
    original_cwd = os.getcwd()
    try:
        os.chdir(TUNER_WORKDIR)
        backend = BenchmarkBackend(os.path.join(original_cwd, "o2tuner"))
        kernels_param_space = {}

        for param_name, spec in tune_config.items():
            if param_name.startswith("PAR_"):
                if spec["type"] == "range":
                    kernels_param_space[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"])
                elif spec["type"] == "values":
                    kernels_param_space[param_name] = trial.suggest_categorical(param_name, spec["values"])
            else:
                grid_value = None
                block_value = None

                if "grid_size" in spec:
                    grid_spec = spec["grid_size"]
                    if grid_spec["type"] == "range":
                        grid_value = trial.suggest_int(f"{param_name}_grid", grid_spec["low"], grid_spec["high"], step=grid_spec.get("step", 1))
                    elif grid_spec["type"] == "values":
                        grid_value = trial.suggest_categorical(f"{param_name}_grid", grid_spec["values"])

                if "block_size" in spec:
                    block_spec = spec["block_size"]
                    if block_spec["type"] == "range":
                        block_value = trial.suggest_int(f"{param_name}_block", block_spec["low"], block_spec["high"], step=block_spec.get("step", 1))
                    elif block_spec["type"] == "values":
                        block_value = trial.suggest_categorical(f"{param_name}_block", block_spec["values"])

                kernels_param_space[param_name] = {}
                if grid_value is not None:
                    kernels_param_space[param_name]["grid_size"] = grid_value
                if block_value is not None:
                    kernels_param_space[param_name]["block_size"] = block_value
        
        for param_name, spec in kernels_param_space.items():
            if isinstance(spec, dict) and "grid_size" in spec and "block_size" in spec:
                min_block_per_cu = spec["grid_size"] / backend.nSMs
                max_threads = spec["block_size"]
                #if min_block_per_cu * max_threads / 256 > 10:
                #    return float("inf")  # Penalize this configuration

        mean, std_dev = backend.get_step_mean_time("optimisation_step", kernels_param_space, "pbpb", "50k", os.path.join(TUNER_WORKDIR, "defaultParamsH100.h"))
    finally:
        os.chdir(original_cwd)
    return mean
