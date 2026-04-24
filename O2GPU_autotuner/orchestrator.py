import os
import time
import sys
from datetime import datetime

class Orchestrator:
    def __init__(self, workdir, steps, backend, header_file):
        self.workdir = workdir
        self.stepTuners = steps 
        self.backend = backend
        self.steps = list(steps.values())
        self.timings = {s.name: None for s in self.steps}     
        self.header_file = header_file
        self.log_file = os.path.join(self.workdir, "tune.log")
        os.makedirs(self.workdir, exist_ok=True)
        self._log(f"===== NEW RUN {datetime.now()} =====")

    def _log(self, msg):
        line = f"[ORCHESTRATOR] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def _collect_params(self, tuners):
        global_config = {}
        for sTuner in tuners:
            global_config.update(sTuner.get_flat_params())
        return global_config

    def _active_tuners(self):
        return [s for s in self.steps if not s.bad_iteration]

    def ask(self):
        for sTuner in self.steps:
            sTuner.ask()
        
    def sample_params(self):
        for sTuner in self.steps:
            sTuner.sample_params()

    def rt_compile_for_HW_stats(self):
        rtc_log = os.path.join(self.workdir, "rtc.log")
        active = self._active_tuners()
        self._log(f"RTC compile with {len(active)} active steps")
        global_config = self._collect_params(active)
        try:
            self.backend.update_param_file(global_config, self.header_file, log_file=self.log_file)
            stats = self.backend.rtc(return_stats=True, log_file=rtc_log)
        except (RuntimeError, TimeoutError) as e:
            self._log(f"RTC compile failed: {e}")
            stats = None
        if stats is not None:
            for sTuner in active:
                sTuner.kernel_stats = stats
            return True

        bad_kernels = self.backend.detectFailingKernels(rtc_log)
        if not bad_kernels:
            self._log("[WARN] Could not detect failing kernels, marking all as bad.")
            for sTuner in active:
                sTuner.bad_iteration = True
            return False

        for kernel_name in bad_kernels:
            self._log(f"Detected failing kernel: {kernel_name}")
            for sTuner in active:
                if kernel_name in sTuner.get_kernel_names():
                    sTuner.update_cache_block_size_limit(kernel_name)
                    sTuner.bad_iteration = True
                    self._log(f"Marking tuner as BAD: {sTuner.name}")

        active = self._active_tuners()
        if not active:
            self._log("[WARN] All tuners marked bad after RTC failure.")
            return False

        global_config = self._collect_params(active)
        try:
            self.backend.update_param_file(global_config, self.header_file, log_file=self.log_file)
            stats = self.backend.rtc(return_stats=True, log_file=rtc_log)
        except (RuntimeError, TimeoutError) as e:
            self._log(f"RTC retry failed: {e}")
            for sTuner in active:
                sTuner.bad_iteration = True
            return False

        for sTuner in active:
            sTuner.kernel_stats = stats
        self._log("RTC retry successful")
        return True
    
    def compute_full_bounds(self):
        active = self._active_tuners()
        ok = True
        for sTuner in active:
            if not sTuner.compute_blocks_per_sm():
                sTuner.bad_iteration = True
                ok = False
        return ok

    def compile_and_run_with_full_bounds(self):
        self.timings = {s.name: None for s in self.steps}
        active = self._active_tuners()
        self._log(f"Benchmark run with {len(active)} active steps")
        global_config = self._collect_params(active)
        try:
            self.backend.update_param_file(global_config, self.header_file, log_file=self.log_file)
            self.backend.profile_benchmark()
        except (RuntimeError, TimeoutError) as e:
            self._log(f"Benchmark failed: {e}")
            return False
                
        for sTuner in self.steps:
            if sTuner.bad_iteration:
                self.timings[sTuner.name] = float("inf")
                continue
            mean, _ = self.backend.compute_step_mean_time(sTuner.name, sTuner.get_flat_params())
            self.timings[sTuner.name] = mean
        self._log("Timing summary:")
        for k, v in self.timings.items():
            self._log(f"  {k}: {v}")
        return True

    def reset_bad_iterations(self):
        for sTuner in self.steps:
            sTuner.bad_iteration = False