import os
import subprocess
import csv
import numpy as np
from collections import defaultdict
import re

class BenchmarkBackend:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.benchmark_backend_log = os.path.join(self.output_folder, 'benchmark_backend.log')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.num_runs = 2
        self.dataset = None
        self.param_dump = "parameters.out"

    def profile_benchmark(self, beamtype=None, IR=None):
        if beamtype is not None and IR is not None:
            dataset = f"o2-{beamtype}-{IR}Hz-32"
        else:
            dataset = self.dataset
        command = [
            "rocprofv2", "--hip-activity", "-d", self.output_folder,
            "-o", "times_raw", "./ca",
            "-e", dataset,
            "--sync", "-g", "--memSize", "15000000000",
            "--preloadEvents", "--runs", str(self.num_runs),
            "--RTCenable", "1", "--RTCTECHloadLaunchBoundsFromFile", self.param_dump
        ]
        timeout = 30 + 45 * self.num_runs # stall timeout check
        with open(self.benchmark_backend_log, 'a') as f:
            try:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=True)
            except subprocess.TimeoutExpired:
                f.write("ERROR: Benchmark stalled and timed out.\n")
                raise TimeoutError("Benchmark timed out")
            except subprocess.CalledProcessError as e:
                f.write(f"ERROR: Benchmark crashed. Return code: {e.returncode}\n")
                raise RuntimeError(f"Benchmark crashed with return code {e.returncode}")

        if os.path.isfile(os.path.join(self.output_folder, 'hip_api_trace_times_raw.csv')):
            os.remove(os.path.join(self.output_folder, 'hip_api_trace_times_raw.csv'))

        hcc_ops_file = os.path.join(self.output_folder, 'hcc_ops_trace_times_raw.csv')
        if os.path.isfile(hcc_ops_file):
            os.rename(hcc_ops_file, os.path.join(self.output_folder, 'times_raw.csv'))

    @staticmethod
    def update_param_file(kernels_config, filename="include/testParam.h", log_file=None):
        macro_updates = []
        kernel_updates = []

        for key, value in kernels_config.items():
            if isinstance(value, dict) and ("block_size" in value or "grid_size" in value):
                kernel_updates.append((key, value))
            elif key.startswith("PAR_"):
                macro_name = f"GPUCA_{key}"
                macro_updates.append((macro_name, value))

        for macro_name, macro_value in macro_updates:
            sed_command = f"sed -E -i 's|^#define {macro_name} .*|#define {macro_name} {macro_value}|' {filename}"
            os.system(sed_command)

        for kernel_name, config in kernel_updates:
            if "block_size" in config:
                block_size = config["block_size"]
                sed_cmd_block = (
                    f"sed -E -i '/^\\s*#define[ \\t]+GPUCA_LB_GPUTPC{kernel_name}[ \\t]+[0-9]+/"
                    f"s/^(\\s*#define[ \\t]+GPUCA_LB_GPUTPC{kernel_name}[ \\t]+)[0-9]+/\\1{block_size}/' {filename}"
                )
                os.system(sed_cmd_block)
            if "grid_size" in config:
                grid_size = config["grid_size"]
                if grid_size % 60 == 0:
                    grid_replacement = f"{grid_size // 60}"
                else:
                    grid_replacement = f"{grid_size // 60}, {grid_size}"
                sed_cmd_grid = (
                    f"sed -E -i '/^\\s*#define[ \\t]+GPUCA_LB_GPUTPC{kernel_name}[ \\t]+[0-9]+/"
                    f"s/^(\\s*#define[ \\t]+GPUCA_LB_GPUTPC{kernel_name}[ \\t]+[0-9]+)(, *[0-9]+)*(, *[0-9]+)*/\\1, {grid_replacement}/' {filename}"
                )
                os.system(sed_cmd_grid)

        root_command = f"echo -e '#define PARAMETER_FILE \"'`pwd`'/{filename}\"\\ngInterpreter->AddIncludePath(\"'`pwd`'/include/GPU\");\\n.x share/GPU/tools/dumpGPUDefParam.C(\"parameters.out\")\\n.q\\n' | root -l -b"
        if log_file is not None:
            with open(log_file, 'a') as f:
                subprocess.run(root_command, shell=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(root_command, shell=True)

    def _compute_durations(self, search_string):
        input_file = os.path.join(self.output_folder, 'times_raw.csv')
        output_file = os.path.join(self.output_folder, f'{search_string}.csv')
        with open(input_file, 'r') as infile:
            reader = csv.DictReader(infile)
            results = []
            for row in reader:
                kernel_name = row["Kernel_Name"]
                if search_string in kernel_name:
                    start_time = int(row["Start_Timestamp"])
                    stop_time = int(row["Stop_Timestamp"])
                    duration = (stop_time - start_time) / 1000000.0
                    results.append({"Kernel_Name": kernel_name, "Duration (ms)": duration})        
        with open(output_file, 'w', newline='') as outfile:
            fieldnames = ["Kernel_Name", "Duration (ms)"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    def _write_stats_to_csv(self, output_file, fieldnames, rows):
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            if not file_exists:
                writer.writerow(fieldnames)
            writer.writerows(rows)

    def _get_run_time_intervals(self):
        input_file = os.path.join(self.output_folder, 'times_raw.csv')
        run_intervals = []
        with open(input_file, 'r') as infile:
            reader = csv.DictReader(infile)
            recording = False
            start_time = None
            for row in reader:
                if row["Operation"] != "KernelExecution":
                    continue
                kernel_name = row["Kernel_Name"]
                if not recording: # skip dummy kernel
                    recording = True
                    continue
                if start_time is None and (not run_intervals or int(row["Start_Timestamp"]) > run_intervals[-1][1]):
                    start_time = int(row["Start_Timestamp"])
                if kernel_name.startswith("krnl_GPUTPCCompressionGatherKernels"):
                    stop_time = int(row["Stop_Timestamp"])
                    run_intervals.append((start_time, stop_time))
                    start_time = None
        return run_intervals

    def _compute_step_duration(self, step_name, kernels_config):
        input_file = os.path.join(self.output_folder, 'times_raw.csv')
        output_file = os.path.join(self.output_folder, f'{step_name}_step.csv')
        step_kernels = set(kernels_config.keys())
        run_intervals = self._get_run_time_intervals()
        step_start_times_per_run = defaultdict(list)
        step_stop_times_per_run = defaultdict(list)
        with open(input_file, 'r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                kernel_name = row["Kernel_Name"].replace("krnl_GPUTPC", "")
                if kernel_name not in step_kernels:
                    continue
                start_time = int(row["Start_Timestamp"])
                stop_time = int(row["Stop_Timestamp"])
                for run_index, (run_start, run_end) in enumerate(run_intervals):
                    if run_start <= start_time <= stop_time <= run_end:
                        step_start_times_per_run[run_index].append(start_time)
                        step_stop_times_per_run[run_index].append(stop_time)
                        break
        results = []
        for run_index in step_start_times_per_run:
            if step_start_times_per_run[run_index]:
                step_start = min(step_start_times_per_run[run_index])
                step_stop = max(step_stop_times_per_run[run_index])
                duration_ms = (step_stop - step_start) / 1e6
                results.append({"Step_Name": step_name, "Duration (ms)": duration_ms})
        with open(output_file, 'w', newline='') as outfile:
            fieldnames = ["Step_Name", "Duration (ms)"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    def _compute_kernel_mean(self, kernel_name, block_size, grid_size, beamtype, IR, write_to_csv=True):
        input_file = os.path.join(self.output_folder, f'{kernel_name}.csv')
        output_file = os.path.join(self.output_folder, f'{kernel_name}_stats.csv')
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)
            data = [float(row[1]) for row in reader]
        mean = np.mean(data)
        std_dev = np.std(data)
        if write_to_csv:
            row = [[block_size, grid_size, beamtype, IR, mean, std_dev]]
            self._write_stats_to_csv(output_file, ["block_size", "grid_size", "beamtype", "IR", "mean", "std_dev"], row)
        return (mean, std_dev)

    def _compute_MergerTrackFit_mean(self, block_size, grid_size, beamtype, IR, write_to_csv=True):
        input_file = os.path.join(self.output_folder, 'GMMergerTrackFit.csv')
        output_file = os.path.join(self.output_folder, 'GMMergerTrackFit_stats.csv')
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)
            data = [float(row[1]) for row in reader]
        results = []
        stats = []
        for i, idx in enumerate(["1", "2"]):
            subset = data[i::2]
            mean = np.mean(subset)
            std_dev = np.std(subset)
            row = [block_size, grid_size, beamtype, IR, idx, mean, std_dev]
            stats.append((mean, std_dev))
            results.append(row)
        if write_to_csv:
            self._write_stats_to_csv(output_file, ["block_size", "grid_size", "beamtype", "IR", "idx", "mean", "std_dev"], results)
        return stats[0]

    def _compute_mean_time(self, kernel_name, block_size, grid_size, beamtype, IR):
        if "GMMergerTrackFit" == kernel_name:
            return self._compute_MergerTrackFit_mean(block_size, grid_size, beamtype, IR)
        else:
            return self._compute_kernel_mean(kernel_name, block_size, grid_size, beamtype, IR)

    def _compute_step_mean_time(self, step_name, kernels_config, beamtype, IR, write_to_csv=True):
        input_file = os.path.join(self.output_folder, f'{step_name}_step.csv')
        output_file = os.path.join(self.output_folder, f'{step_name}_step_stats.csv')
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)
            data = [float(row[1]) for row in reader]
        mean = np.mean(data)
        std_dev = np.std(data)
        if write_to_csv:
            row = [[kernels_config, beamtype, IR, mean, std_dev]]
            self._write_stats_to_csv(output_file, ["kernels_config", "beamtype", "IR", "mean", "std_dev"], row)
        return (mean, std_dev)
    
    def get_kernel_mean_time(self, kernel_name, block_size, grid_size, beamtype, IR):
        kernels_config = {kernel_name: {"block_size": block_size, "grid_size": grid_size}}
        BenchmarkBackend.update_param_file(kernels_config, log_file=self.benchmark_backend_log)
        try:
            self.profile_benchmark(beamtype, IR)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error during benchmark: {e}")
            return (float('inf'), 0.0)
        self._compute_durations(kernel_name)
        return self._compute_mean_time(kernel_name, block_size, grid_size, beamtype, IR)

    def get_step_mean_time(self, step_name, kernels_config, beamtype=None, IR=None):
        BenchmarkBackend.update_param_file(kernels_config, log_file=self.benchmark_backend_log)
        try:
            self.profile_benchmark(beamtype, IR)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error during step benchmark: {e}")
            return (float('inf'), 0.0)
        for kernel_name in kernels_config.keys():
            self._compute_durations(kernel_name)
        self._compute_step_duration(step_name, kernels_config)
        return self._compute_step_mean_time(step_name, kernels_config, beamtype, IR)
    
    def get_sync_mean_time(self, dump=None, beamtype=None, IR=None, dataset=None):
        self.param_dump = dump or self.param_dump or "param_dumps/default.par"
        dataset = dataset or (f"o2-{beamtype}-{IR}Hz-32" if beamtype and IR else self.dataset)
        command = [
            "./ca", "-e", dataset,
            "--sync", "-g", "--memSize", "15000000000",
            "--preloadEvents", "--runs", str(self.num_runs),
            "--RTCenable", "1", "--RTCTECHloadLaunchBoundsFromFile", self.param_dump
        ]
        timeout = 30 + 45 * self.num_runs  # stall timeout check
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
            output = result.stdout
        except subprocess.TimeoutExpired:
            with open(self.benchmark_backend_log, 'a') as f:
                f.write("ERROR: Benchmark stalled and timed out.\n")
            raise TimeoutError("Benchmark timed out")
        except subprocess.CalledProcessError as e:
            with open(self.benchmark_backend_log, 'a') as f:
                f.write(f"ERROR: Benchmark crashed. Return code: {e.returncode}\n")
            raise RuntimeError(f"Benchmark crashed with return code {e.returncode}")
        
        with open(self.benchmark_backend_log, 'a') as f:
            f.write(output)

        wall_times = re.findall(r"Total Wall Time:\s+([0-9]+) us", output)
        wall_times = [int(x) for x in wall_times]
        if not wall_times:
            return float('inf'), 0.0
        wall_times_ms = np.array(wall_times) / 1000.0 
        return np.mean(wall_times_ms), np.std(wall_times_ms)
