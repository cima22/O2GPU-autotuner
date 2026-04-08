import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TUNE_SPACE_PATH = os.getenv("TUNE_SPACE_PATH", os.path.join(os.path.dirname(__file__), "tune_space.yaml"))
TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../standalone"))
TUNE_SPACE_NAME = os.getenv("TUNE_SPACE_NAME")
TUNER_DATASET = os.getenv("TUNER_DATASET", "o2-pbpb-47kHz-32")
TUNER_PARAMETER_FILE = os.getenv("TUNER_PARAMETER_FILE", os.path.join(TUNER_WORKDIR, "defaultParams.h"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "tuning_results"))

with open(TUNE_SPACE_PATH, "r") as f:
    tune_config = yaml.safe_load(f)

from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend

def optimise(trial):
    original_cwd = os.getcwd()
    try:
        os.chdir(TUNER_WORKDIR)
        backend = BenchmarkBackend(OUTPUT_DIR)
        kernels_param_space = {}

        for param_name, spec in tune_config.items():
            if param_name.startswith("PAR_"):
                if spec["type"] == "range":
                    kernels_param_space[param_name] = trial.suggest_int(param_name, spec["min"], spec["max"])
                elif spec["type"] == "values":
                    kernels_param_space[param_name] = trial.suggest_categorical(param_name, spec["values"])
            else:
                blocks_per_sm = None
                block_value = None

                if "blocks_per_sm" in spec:
                    blocks_per_sm_spec = spec["blocks_per_sm"]
                    if blocks_per_sm_spec["type"] == "range":
                        blocks_per_sm = trial.suggest_int(f"{param_name}_blocks_per_sm", blocks_per_sm_spec["min"], blocks_per_sm_spec["max"], step=blocks_per_sm_spec.get("step", 1))
                    elif blocks_per_sm_spec["type"] == "values":
                        blocks_per_sm = trial.suggest_categorical(f"{param_name}_blocks_per_sm", blocks_per_sm_spec["values"])

                if "block_size" in spec:
                    block_spec = spec["block_size"]
                    if block_spec["type"] == "range":
                        block_value = trial.suggest_int(f"{param_name}_block", block_spec["min"] * backend.warpSize, block_spec["max_value"], step=block_spec.get("step", 1) * backend.warpSize)
                    elif block_spec["type"] == "values":
                        warp_values = [v for v in block_spec["values"] if v % backend.warpSize == 0]
                        block_value = trial.suggest_categorical(f"{param_name}_block", warp_values)

                kernels_param_space[param_name] = {}
                if blocks_per_sm is not None:
                    kernels_param_space[param_name]["blocks_per_sm"] = blocks_per_sm
                if block_value is not None:
                    kernels_param_space[param_name]["block_size"] = block_value
        
        for param_name, spec in kernels_param_space.items():
            if isinstance(spec, dict) and "blocks_per_sm" in spec and "block_size" in spec:
                min_block_per_cu = spec["blocks_per_sm"]
                max_threads = spec["block_size"]
                if (min_block_per_cu * max_threads > backend.maxThreadsPerMultiProcessor) and (TUNE_SPACE_NAME != "tracklet"):
                    return float("inf")  # Penalize this configuration

        mean, std_dev = backend.get_step_mean_time("optimisation_step", kernels_param_space, dataset=TUNER_DATASET, filename=TUNER_PARAMETER_FILE)
    finally:
        os.chdir(original_cwd)
    return mean
