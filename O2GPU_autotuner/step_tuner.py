# step_tuner.py
import optuna
from dataclasses import dataclass, field
from typing import Optional

class StepTuner:
    def __init__(self, name: str, tune_config: dict, backend, output_dir: str, startup: int):
        self.name = name
        self.config = tune_config
        self.backend = backend
        self.study = optuna.create_study(study_name=name, direction="minimize", sampler=optuna.samplers.TPESampler(n_startup_trials=startup, multivariate=True), storage=f"sqlite:///{output_dir}/{name}.db", load_if_exists=True)
        self.trial = None
        self.params = {}   # param_name -> {block_size, blocks_per_sm}
        self.par_params = {}   # PAR_* -> value
        self.bad_iteration = False
        self.cache_block_size_limit = {}

    def ask(self):
        self.trial  = self.study.ask()
        self.params = {}
        self.par_params = {}

    def tell(self, value: float):
        self.study.tell(self.trial, value)

    def sample_params(self):
        for param_name, spec in self.config.items():
            if param_name.startswith("PAR_"):
                self.par_params[param_name] = self._sample_par(param_name, spec)
                continue
            block_size, min_blocks_per_sm = self._sample_launch_bounds(param_name, spec)
            self.params[param_name] = {"block_size": block_size, "blocks_per_sm": min_blocks_per_sm}

    def get_flat_params(self) -> dict:
        merged = dict(self.par_params)
        merged.update(self.params)
        return merged

    def get_kernel_names(self) -> list:
        return list(self.params.keys())

    @property
    def best_trial(self):
        return self.study.best_trial

    def _sample_par(self, param_name, spec):
        if spec["type"] == "range":
            return self.trial.suggest_int(param_name, spec["min"], spec["max"])
        elif spec["type"] == "values":
            return self.trial.suggest_categorical(param_name, spec["values"])
        elif spec["type"] == "block_size_range":
            effective_max = spec.get("max_value")
            if effective_max == "max_block_size":
                effective_max = self.backend.GPUlimits["max_threads_per_block"]
                if effective_max < spec["min"] * self.backend.warpSize:
                    raise optuna.TrialPruned()
            return self.trial.suggest_int(param_name, spec["min"] * self.backend.warpSize, effective_max, step=spec.get("step", 1) * self.backend.warpSize)

    def _register_kernel_attrs(self, param_name, block_size, blocks_per_sm=1, max_bpsm=1):
        self.trial.set_user_attr(f"{param_name}_block_size", block_size)
        self.trial.set_user_attr(f"{param_name}_blocks_per_sm", blocks_per_sm)
        self.trial.set_user_attr(f"{param_name}_max_bpsm", max_bpsm)

    def _sample_launch_bounds(self, param_name, spec):
        bs_spec = spec.get("block_size")
        if bs_spec is None:
            raise ValueError(f"block_size not defined for {param_name}")
        cached_max = self.cache_block_size_limit.get(param_name)
        if bs_spec["type"] == "single":
            block_size = bs_spec["values"][self.backend.backend]
            block_size = (block_size // self.backend.warpSize) * self.backend.warpSize
            self._register_kernel_attrs(param_name, block_size, blocks_per_sm=1, max_bpsm=1)
            return block_size
        elif bs_spec["type"] == "range":
            if bs_spec["type"] == "max_block_size":
                effective_max = self.backend.GPUlimits["max_threads_per_block"]
            if cached_max is not None:
                effective_max = min(effective_max, cached_max)
            if effective_max < bs_spec["min"] * self.backend.warpSize:
                raise optuna.TrialPruned()
            block_size = self.trial.suggest_int(f"{param_name}_block_size", bs_spec["min"] * self.backend.warpSize, effective_max, step=bs_spec.get("step", 1) * self.backend.warpSize)
        elif bs_spec["type"] == "values":
            warp_values = [v for v in bs_spec["values"] if v % self.backend.warpSize == 0]
            if cached_max is not None:
                warp_values = [v for v in warp_values if v <= cached_max]
            if not warp_values:
                raise optuna.TrialPruned()
            block_size = self.trial.suggest_categorical(f"{param_name}_block_size", warp_values)
        block_size = (block_size // self.backend.warpSize) * self.backend.warpSize
        min_blocks_per_sm, max_bpsm = self._sample_kernel_bpsm(block_size)
        self._register_kernel_attrs(param_name, block_size, blocks_per_sm=min_blocks_per_sm, max_bpsm=max_bpsm)
        return block_size, min_blocks_per_sm

    def _sample_kernel_bpsm(self, block_size):
        lim  = self.backend.GPUlimits
        max_bpsm_threads = lim["max_threads_per_sm"] // block_size
        max_bpsm_hw      = lim["max_blocks_per_sm"]
        max_bpsm         = min(max_bpsm_threads, max_bpsm_hw)
        if max_bpsm < 1:
            return None, None
        fraction = self.trial.suggest_float(f"{param_name}_blocks_per_sm_fraction", 0.0, 1.0)
        blocks_per_sm = min(max_bpsm, int(math.floor(fraction * (max_bpsm + 1))));
        return blocks_per_sm, max_bpsm

    def update_cache_block_size_limit(self, kernel_name):
        current_block_size = self.params.get(kernel_name, {}).get("block_size")
        if self.cache_block_size_limit.get(kernel_name) is None:
            self.cache_block_size_limit[kernel_name] = current_block_size
        else:
            self.cache_block_size_limit[kernel_name] = min(self.cache_block_size_limit[kernel_name], current_block_size)