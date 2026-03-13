import os
import subprocess
import csv
import numpy as np
from collections import defaultdict
import re
import shutil
import pandas as pd
import glob
import shutil
import json

class BenchmarkBackend:
    def __init__(self, output_folder, backend="auto", debug=False):
        self.output_folder = output_folder
        self.benchmark_backend_log = os.path.join(self.output_folder, 'benchmark_backend.log')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.num_runs = 3
        self.dataset = None
        self.param_dump = "parameters.out"
        self.debug = debug
        self.backend = backend if backend != "auto" else BenchmarkBackend._detect_GPUs_vendor(self.debug)
        if self.backend not in ["amd", "nvidia"]:
            print("Warning: Unsupported or unknown GPU backend detected")
            return
        if self.backend == "amd":
            self.profiler = "rocprofv2"
            self.profiler_options = ["--hip-activity",  "-o", "times_raw", "-d", f"{self.output_folder}"]
            self.gpu_lang = "HIP"
            self.nSMs = BenchmarkBackend._AMD_get_number_of_compute_units()
            self._get_df_from_raw = self._AMD_get_df_from_raw
        if self.backend == "nvidia":
            self.profiler = "nsys"
            self.profiler_options = ["profile", "-o", f"{os.path.join(self.output_folder, 'report.nsys-rep')}", "--force-overwrite", "true"]
            self.gpu_lang = "CUDA"
            self.nSMs = BenchmarkBackend._NVIDIA_get_number_of_streaming_multiprocessors()
            self._get_df_from_raw = self._NVIDIA_get_df_from_raw
        try:
            BenchmarkBackend._detect_profiler(self.profiler)
        except FileNotFoundError as e:
            print("Profiler not found! Please ensure it is installed and in your PATH.")
            
    @staticmethod
    def _detect_GPUs_vendor(debug=False):
        if shutil.which("nvidia-smi"):
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                if debug:
                    print("Detected NVIDIA GPU(s)")
                return "nvidia"
            except subprocess.CalledProcessError:
                print("nvidia-smi found but not working properly.")
        if shutil.which("rocm-smi"):
            try:
                subprocess.run(["rocm-smi"], capture_output=True, text=True, check=True)
                if debug:
                    print("Detected AMD GPU(s)")
                return "amd"
            except subprocess.CalledProcessError:
                print("rocm-smi found but not working properly.")
        return None

    #@staticmethod
    #def _detect_GPUs_vendor():
    #        "0x10de": "nvidia",
    #    VENDOR_MAP = {
    #        "0x1002": "amd",
    #        "0x8086": "intel",
    #    }
    #    vendors = []
    #    for vfile in glob.glob("/sys/class/drm/card*/device/vendor"):
    #        if not os.access(vfile, os.R_OK):
    #            continue
    #        try:
    #            with open(vfile) as f:
    #                vid = f.read().strip().lower()
    #            if vid in VENDOR_MAP:
    #                vendors.append(VENDOR_MAP[vid])
    #            else:
    #                vendors.append(f"unknown({vid})")
    #        except Exception as e:
    #            print(f"Skipping {vfile}: {e}")
    #    
    #    for vendor in ["nvidia", "amd"]:
    #        if vendor in vendors:
    #            print(f"Detected {vendor.upper()} GPU(s)")
    #            return vendor
    #    print("Detected multiple GPU vendors:", vendors)
    #    print("Using the first one:", vendors[0])
    #    return vendors[0]
        
    @staticmethod
    def _AMD_get_number_of_compute_units():
        cmd = "rocminfo | grep -A15 'GPU' | grep 'Compute Unit' | head -n1 | awk '{print $3}'"
        cmd = "echo '#include <hip/hip_runtime.h>\n#include <stdio.h>\nint main(){hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf(\"%d\\n\",p.multiProcessorCount);return 0;}' > /tmp/sm.cpp && hipcc /tmp/sm.cpp -o /tmp/sm && /tmp/sm"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            nCU = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            nCU = 1
        return nCU
    
    @staticmethod
    def _NVIDIA_get_number_of_streaming_multiprocessors():
        binary_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../count_SM_Nvidia")
        try:
            result = subprocess.run(binary_abs_path, shell=True, capture_output=True, text=True, check=True)
            nSM = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            nSM = 1
        return nSM

    @staticmethod
    def _detect_profiler(profiler):
        try:
            subprocess.run([profiler, "--version"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(f"{profiler} not found or not working properly.") from e
        
    def profile_benchmark(self, dataset=None, dump=None):
        self.dataset = dataset or self.dataset
        rtc_dump = ["./ca", "--noEvents", "--sync", "-g", "--gpuType", self.gpu_lang, "--memSize", "15000000000", "--RTCenable", "1", "--RTCcacheOutput", "1", "--RTCTECHrunTest", "2"]
        command = [self.profiler] + self.profiler_options
        command += ["./ca", "-e", self.dataset, "--sync", "-g", "--gpuType", self.gpu_lang, "--memSize", "15000000000", "--preloadEvents", "-s", "0", "-r", "3", "-n", str(self.num_runs), "--RTCenable", "1", "--RTCcacheOutput", "1"]
        if dump:
            dump_command = ["--RTCTECHloadLaunchBoundsFromFile", self.param_dump]
            rtc_dump += dump_command
            command += dump_command
        timeout = 90
        with open(self.benchmark_backend_log, 'a') as f:
            try:
                if self.backend == "nvidia": #creating dump for nvidia otherwise timeline dumping takes ages
                    subprocess.run(rtc_dump, stdout=f, stderr=f, timeout=timeout, check=True)
                timeout = 60 * self.num_runs # stall timeout check
                subprocess.run(command, stdout=f, stderr=f, timeout=timeout, check=True)
            except subprocess.TimeoutExpired:
                f.write("ERROR: Benchmark stalled and timed out.\n")
                raise TimeoutError("Benchmark timed out")
            except subprocess.CalledProcessError as e:
                f.write(f"ERROR: Benchmark crashed. Return code: {e.returncode}\n")
                raise RuntimeError(f"Benchmark crashed with return code {e.returncode}")
        self._postprocess_profiler_output()

    def _postprocess_profiler_output(self):
        if self.backend == "amd":
            if os.path.isfile(os.path.join(self.output_folder, 'hip_api_trace_times_raw.csv')):
                os.remove(os.path.join(self.output_folder, 'hip_api_trace_times_raw.csv'))
            hcc_ops_file = os.path.join(self.output_folder, 'hcc_ops_trace_times_raw.csv')
            if os.path.isfile(hcc_ops_file):
                os.rename(hcc_ops_file, os.path.join(self.output_folder, 'times_raw.csv'))
        if self.backend == "nvidia":
            report_file = os.path.join(self.output_folder, "report.nsys-rep")
            json_file = os.path.join(self.output_folder, "times_raw.json")
            if os.path.isfile(report_file):
                command = f"nsys export --type json --output {json_file} --separate-strings 1 --include-json 1 -f 1 {report_file}"
                subprocess.run(command, shell=True, cwd=self.output_folder)

    def _AMD_get_df_from_raw(self):
        input_file = os.path.join(self.output_folder, 'times_raw.csv')
        stats = pd.read_csv(input_file)
        stats = stats[stats["Operation"] == "KernelExecution"]
        stats = stats[["Kernel_Name", "Start_Timestamp", "Stop_Timestamp"]]
        stats["Start_Timestamp"] = pd.to_numeric(stats["Start_Timestamp"], errors='coerce')
        stats["Stop_Timestamp"] = pd.to_numeric(stats["Stop_Timestamp"], errors='coerce')
        stats["DurationMs"] = (stats["Stop_Timestamp"] - stats["Start_Timestamp"]) / 1e6
        stats.rename(columns={'Start_Timestamp': 'StartNs', 'Stop_Timestamp': 'EndNs'}, inplace=True)
        return stats
    
    def _NVIDIA_get_df_from_raw(self):
        input_file = os.path.join(self.output_folder, 'times_raw.json')
        names = []
        events = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        if row.get("type") == "String":
                            names.append(row)
                        elif row.get("Type") == 79:
                            cuda = row.get("CudaEvent", {})
                            kernel = cuda.get("kernel", {})
                            row["demangledName"] = kernel.get("demangledName")
                            row["startNs"] = cuda.get("startNs")
                            row["endNs"] = cuda.get("endNs")
                            events.append(row)
                except json.JSONDecodeError:
                    print("Skipping malformed line:", line[:80])

        names = pd.DataFrame(names)
        events = pd.DataFrame(events)
        merged = pd.merge(names, events, left_on="id", right_on="demangledName", how="inner")
        merged = merged[["value", "startNs", "endNs"]]
        merged["startNs"] = pd.to_numeric(merged["startNs"], errors='coerce')
        merged["endNs"] = pd.to_numeric(merged["endNs"], errors='coerce')
        merged["DurationMs"] = (merged["endNs"] - merged["startNs"]) / 1e6
        merged.rename(columns={'value': 'Kernel_Name', 'startNs': 'StartNs', 'endNs': 'EndNs'}, inplace=True)
        return merged

    def _get_views(self):
        kernel_stats = self._get_df_from_raw()
        sorted_by_start = kernel_stats.sort_values(by="StartNs").reset_index(drop=True)
        last_kernel = "krnl_GPUTPCCompressionGatherKernels_multiBlock"
        indices = sorted_by_start.index[sorted_by_start["Kernel_Name"] == last_kernel].tolist()
        boundaries = [-1] + indices + [len(sorted_by_start)]
        views = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i] + 1
            end = boundaries[i+1] + 1
            view = sorted_by_start.iloc[start:end]
            if not view.empty:
                views.append(view)
        self.views = views
        return views

    def _compute_kernel_mean_time(self, kernel_name):
        kernel_stats = self._get_df_from_raw()
        filtered = kernel_stats[kernel_stats["Kernel_Name"].str.contains(kernel_name)]
        if filtered.empty:
            return float('inf'), 0.0
        data = filtered["DurationMs"].to_numpy()
        mean = np.mean(data)
        std_dev = np.std(data)
        return (mean, std_dev)

    def _compute_step_mean_time(self, step_kernels):
        subviews = []
        durations = []
        self._get_views()
        for view in self.views:
            subview = view[view["Kernel_Name"].str.contains('|'.join(step_kernels))]
            if not subview.empty:
                subviews.append(subview)
        for subview in subviews:
            earliest_start = subview["StartNs"].min()
            latest_stop = subview["EndNs"].max()
            duration_ms = (latest_stop - earliest_start) / 1e6
            durations.append(duration_ms)
        data = np.array(durations)
        mean = np.mean(data)
        std_dev = np.std(data)
        return (mean, std_dev)
    
    def get_kernel_mean_time(self, kernel_name, dataset=None, dump=None):
        try:
            self.profile_benchmark(dataset, dump)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error during benchmark: {e}")
            return (float('inf'), 0.0)
        return self._compute_kernel_mean_time(kernel_name)

    def get_step_mean_time(self, step_kernels, dataset=None, dump=None):
        dataset = dataset or self.dataset
        try:
            self.profile_benchmark(dataset, dump)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error during step benchmark: {e}")
            return (float('inf'), 0.0)
        return self._compute_step_mean_time(step_kernels)
    
    def get_sync_mean_time(self, dataset = None, dump=None):
        dataset = dataset or self.dataset
        command = [
            "./ca", "-e", dataset,
            "--sync", "-g", "--gpuType", self.gpu_lang, "--memSize", "15000000000",
            "--preloadEvents", "-n", str(self.num_runs),
            "--RTCenable", "1"]
        if dump:
            command += ["--RTCTECHloadLaunchBoundsFromFile", dump]
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
