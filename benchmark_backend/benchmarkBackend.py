import os
import subprocess
import csv
import numpy as np
from collections import defaultdict

class BenchmarkBackend:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def profile_benchmark(self, beamtype, IR):
        command = [
            "rocprofv2", "--hip-activity", "-d", self.output_folder, 
            "-o", "times_raw", "./ca", 
            "-e", f"o2-{beamtype}-{IR}Hz-128", 
            "--sync", "-g", "--memSize", "30000000000", 
            "--preloadEvents", "--runs", "2", 
            "--RTCdeterministic", "1", "--RTCenable", "1", 
            "--RTCcacheOutput", "1", "--RTCTECHloadLaunchBoundsFromFile", "parameters.out"
        ]
        log_file = os.path.join(self.output_folder, 'time_kernels.log')
        with open(log_file, 'a') as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if os.path.exists(self.output_folder):
            if os.path.isfile(os.path.join(self.output_folder, 'hip_api_trace_times_raw.csv')):
                os.remove(os.path.join(self.output_folder, 'hip_api_trace_times_raw.csv'))

            hcc_ops_file = os.path.join(self.output_folder, 'hcc_ops_trace_times_raw.csv')
            if os.path.isfile(hcc_ops_file):
                os.rename(hcc_ops_file, os.path.join(self.output_folder, 'times_raw.csv'))
        else:
            raise FileNotFoundError(f"The directory '{self.output_folder}' does not exist.")

    def update_param_file(self, kernels_config, filename="include/testParam.h"):
        for kernel_name, config in kernels_config.items():
            block_size = config["block_size"]
            grid_size = config["grid_size"]
            sed_command = (
                f"sed -i '/^\\s*#define GPUCA_LB_GPUTPC{kernel_name} \\s*[0-9][0-9]*,\\? *[0-9]*/"
                f"{{s/\\s[0-9][0-9]*,\\? *[0-9]*/ {block_size}, {grid_size//60}/}}' {filename}")
            os.system(sed_command)
        root_command = "root -l -q -b src/GPU/GPUTracking/Standalone/tools/dumpGPUDefParam.C"
        log_file = os.path.join(self.output_folder, 'time_kernels.log')
        with open(log_file, 'a') as f:
            subprocess.run(root_command, shell=True, stdout=f, stderr=subprocess.STDOUT)

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
                if start_time is None:
                    start_time = int(row["Start_Timestamp"])
                if kernel_name.startswith("krnl_GPUTPCCompressionGatherKernels"):
                    stop_time = int(row["Stop_Timestamp"])
                    run_intervals.append((start_time, stop_time))
                    start_time = None
        return run_intervals

    def _compute_step_duration(self, step_name, kernels_config):
        input_file = os.path.join(self.output_folder, 'times_raw.csv')
        output_file = os.path.join(self.output_folder, f'{step_name}.csv')
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

    def _compute_MergerCollect_mean(self, block_size, grid_size, beamtype, IR, write_to_csv=True):
        input_file = os.path.join(self.output_folder, 'GMMergerCollect.csv')
        output_file = os.path.join(self.output_folder, 'GMMergerCollect_stats.csv')
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)
            data = [float(row[1]) for row in reader]
        results = []
        stats = []
        for i, proc_type in enumerate(["sync", "async"]):
            subset = data[i::2]
            mean = np.mean(subset)
            std_dev = np.std(subset)
            row = [block_size, grid_size, beamtype, IR, proc_type, mean, std_dev]
            stats.append((mean, std_dev))
            results.append(row)
        if write_to_csv:
            self._write_stats_to_csv(output_file, ["block_size", "grid_size", "beamtype", "IR", "proc_type", "mean", "std_dev"], results)
        return stats[0]

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
        input_file = os.path.join(self.output_folder, f'{step_name}.csv')
        output_file = os.path.join(self.output_folder, f'{step_name}_stats.csv')
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
        kernles_config = {kernel_name: {"block_size": block_size, "grid_size": grid_size}}
        self.update_param_file(kernles_config)
        self.profile_benchmark(beamtype, IR)
        self._compute_durations(kernel_name)        
        return self._compute_mean_time(kernel_name, block_size, grid_size, beamtype, IR)

    def get_step_mean_time(self, step_name, kernels_config, beamtype, IR):
        self.update_param_file(kernels_config)
        self.profile_benchmark(beamtype, IR)
        for kernel_name in kernels_config.keys():
            self._compute_durations(kernel_name)
        self._compute_step_duration(step_name, kernels_config)
        return self._compute_step_mean_time(step_name, kernels_config, beamtype, IR)
