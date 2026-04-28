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
            block_size = self._sample_block_size(param_name, spec)
            self.params[param_name] = {"block_size": block_size, "blocks_per_sm": 1}
            self.trial.suggest_float(f"{param_name}_blocks_per_sm_fraction", 0.0, 1.0)

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
        elif spec["block_size_range"]:
            return self.trial.suggest_int(param_name, spec["min"] * self.backend.warpSize, spec["max_value"], step=spec.get("step", 1) * self.backend.warpSize)

    def _register_kernel_attrs(self, param_name, block_size, blocks_per_sm=1, max_bpsm=1):
        self.trial.set_user_attr(f"{param_name}_block_size", block_size)
        self.trial.set_user_attr(f"{param_name}_blocks_per_sm", blocks_per_sm)
        self.trial.set_user_attr(f"{param_name}_max_bpsm", max_bpsm)

    def _sample_block_size(self, param_name, spec):
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
            effective_max = min(bs_spec["max_value"], cached_max) if cached_max else bs_spec["max_value"]
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
        self._register_kernel_attrs(param_name, block_size, blocks_per_sm=1, max_bpsm=1)
        return block_size

    def compute_blocks_per_sm(self):
        for param_name, v in self.params.items():
            stats = self.kernel_stats.get(param_name)
            if stats is None:
                self.params[param_name]["blocks_per_sm"] = 1
                self.trial.set_user_attr(f"{param_name}_blocks_per_sm", 1)
                self.trial.set_user_attr(f"{param_name}_max_bpsm", 1)
                continue
            bpsm = self._compute_kernel_bpsm(param_name)
            if bpsm is None:
                return False
            self.params[param_name]["blocks_per_sm"] = bpsm
        return True

    def _compute_kernel_bpsm(self, param_name) -> Optional[int]:
        shm  = self.kernel_stats[param_name]["shared_memory"]
        regs = self.kernel_stats[param_name]["registers"]
        regs = -1
        lim  = self.backend.GPUlimits
        block_size = self.params[param_name]["block_size"]
        max_bpsm_threads = lim["max_threads_per_sm"] // block_size
        max_bpsm_hw      = lim["max_blocks_per_sm"]
        max_bpsm_regs    = lim["registers_per_sm"] // (regs * block_size) if regs > 0 else max_bpsm_hw
        max_bpsm_shm     = lim["shared_mem_per_sm"] // shm if shm > 0 else max_bpsm_hw
        max_bpsm         = min(max_bpsm_threads, max_bpsm_hw, max_bpsm_regs, max_bpsm_shm)
        if max_bpsm < 1:
            return None
        fraction = self.trial.params[f"{param_name}_blocks_per_sm_fraction"]
        blocks_per_sm = max(1, round(fraction * max_bpsm))
        self.trial.set_user_attr(f"{param_name}_blocks_per_sm", blocks_per_sm)
        self.trial.set_user_attr(f"{param_name}_max_bpsm", max_bpsm)
        return blocks_per_sm

    def update_cache_block_size_limit(self, kernel_name):
        current_block_size = self.params.get(kernel_name, {}).get("block_size")
        if self.cache_block_size_limit.get(kernel_name) is None:
            self.cache_block_size_limit[kernel_name] = current_block_size
        else:
            self.cache_block_size_limit[kernel_name] = min(self.cache_block_size_limit[kernel_name], current_block_size)