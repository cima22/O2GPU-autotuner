from fileinput import filename
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
from pathlib import Path
import re

class BenchmarkBackend:
    def __init__(self, output_folder, backend="auto", debug=False):
        self.output_folder = output_folder
        self.benchmark_backend_log = os.path.join(self.output_folder, 'benchmark_backend.log')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.num_runs = 3
        self.num_events = None
        self.dataset = None
        self.param_dump = "parameters.out"
        self.debug = debug
        self.vRAM = 13000000000
        self.backend = backend if backend != "auto" else BenchmarkBackend._detect_GPUs_vendor(self.debug)
        self._kernel_hw_stats_cache = {}
        if os.path.exists("rtccache"):
            shutil.rmtree("rtccache", ignore_errors=True)
        if self.backend not in ["amd", "nvidia"]:
            print("Warning: Unsupported or unknown GPU backend detected")
            return
        if self.backend == "amd":
            self.profiler = "rocprofv2"
            self.profiler_options = ["--hip-activity",  "-o", "times_raw", "-d", f"{self.output_folder}"]
            self.gpu_lang = "HIP"
            self.warpSize = BenchmarkBackend._AMD_get_warp_size()
            self.nSMs = BenchmarkBackend._AMD_get_number_of_compute_units()
            self.GPUlimits = BenchmarkBackend._AMD_get_sm_limits()
            self._get_df_from_raw = self._AMD_get_df_from_raw
            self.detectFailingKernels = self._AMD_detectFailingKernels
            self.get_rtc_HW_stats = self._AMD_get_rtc_HW_stats
        if self.backend == "nvidia":
            self.profiler = "nsys"
            self.profiler_options = ["profile", "-o", f"{os.path.join(self.output_folder, 'report.nsys-rep')}", "--force-overwrite", "true"]
            self.gpu_lang = "CUDA"
            self.warpSize = BenchmarkBackend._NVIDIA_get_warp_size()
            self.nSMs = BenchmarkBackend._NVIDIA_get_number_of_streaming_multiprocessors()
            self.GPUlimits = BenchmarkBackend._NVIDIA_get_sm_limits()
            self._get_df_from_raw = self._NVIDIA_get_df_from_raw
            self.detectFailingKernels = self._NVIDIA_detectFailingKernels
            self.get_rtc_HW_stats = self._NVIDIA_get_rtc_HW_stats
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
    def _AMD_get_warp_size():
        cmd = "echo '#include <hip/hip_runtime.h>\n#include <stdio.h>\nint main(){hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf(\"%d\\n\",p.warpSize);return 0;}' > /tmp/warp.cpp && hipcc /tmp/warp.cpp -o /tmp/warp && /tmp/warp"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            warp_size = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            warp_size = 64
        return warp_size

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
    def _AMD_get_max_threads_per_cu():
        cmd = "echo '#include <hip/hip_runtime.h>\n#include <stdio.h>\nint main(){hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf(\"%d\\n\",p.maxThreadsPerMultiProcessor);return 0;}' > /tmp/maxThreads.cpp && hipcc /tmp/maxThreads.cpp -o /tmp/maxThreads && /tmp/maxThreads"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            maxThreadsPerMultiProcessor = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            maxThreadsPerMultiProcessor = 2560
        return maxThreadsPerMultiProcessor

    @staticmethod
    def _AMD_get_kernel_HW_stats():
        pass
    
    @staticmethod
    def _AMD_get_sm_limits():
        pass

    @staticmethod
    def _NVIDIA_get_warp_size():
        return 32
    
    @staticmethod
    def _NVIDIA_get_number_of_streaming_multiprocessors():
        cmd = "echo '#include <cuda_runtime.h>\n#include <stdio.h>\nint main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf(\"%d\\n\",p.multiProcessorCount);return 0;}' > /tmp/count_sm.cu && nvcc /tmp/count_sm.cu -o /tmp/count_sm && /tmp/count_sm"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            nSM = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            nSM = 1
        return nSM
    
    @staticmethod
    def _NVIDIA_get_max_threads_per_sm():
        cmd = "echo '#include <cuda_runtime.h>\n#include <stdio.h>\nint main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf(\"%d\\n\",p.maxThreadsPerMultiProcessor);return 0;}' > /tmp/max_threads.cu && nvcc /tmp/max_threads.cu -o /tmp/max_threads && /tmp/max_threads"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            max_threads = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            max_threads = 2048
        return max_threads

    @staticmethod
    def _NVIDIA_get_sm_limits():
        cmd = ("echo '#include <cuda_runtime.h>\n#include <stdio.h>\nint main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);"
            "printf(\"%d %d %d %d\\n\",p.maxThreadsPerMultiProcessor,p.regsPerMultiprocessor,p.sharedMemPerBlock,p.maxBlocksPerMultiProcessor);return 0;}' "
            "> /tmp/sm_limits.cu && nvcc /tmp/sm_limits.cu -o /tmp/sm_limits && /tmp/sm_limits")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            max_threads, regs, shm, max_blocks = map(int, result.stdout.strip().split())
        except (subprocess.CalledProcessError, ValueError):
            max_threads, regs, shm, max_blocks = 2048, 65536, 49152, 32
        return {"max_threads_per_sm": max_threads, "registers_per_sm": regs, "shared_mem_per_sm": shm, "max_blocks_per_sm": max_blocks}

    @staticmethod
    def _NVIDIA_detectFailingKernels(log_file):
        if not os.path.exists(log_file):
            return set()
        with open(log_file, "r") as f:
            content = f.read()
        if "ptxas error" not in content:
            return set()
        matches = re.findall(r"ptxas error.*Entry function 'krnl_GPUTPC(\w+)'", content, re.DOTALL)
        return set(matches)
    
    @staticmethod
    def _NVIDIA_get_compute_capability():
        cmd = "echo '#include <cuda_runtime.h>\n#include <stdio.h>\nint main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf(\"%d%d\\n\",p.major,p.minor);return 0;}' > /tmp/get_cc.cu && nvcc /tmp/get_cc.cu -o /tmp/get_cc && /tmp/get_cc"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return result.stdout.strip()      
        except (subprocess.CalledProcessError, ValueError):
            return None

    @staticmethod
    def _NVIDIA_get_rtc_HW_stats(fatbin_files=None):
        CC = BenchmarkBackend._NVIDIA_get_compute_capability()
        arch = f"sm_{CC}" if CC else None
        if fatbin_files is None:
            fatbin_files = glob.glob("/tmp/o2cagpu_rtc_*.fatbin")
        if not fatbin_files:
            return {}
        stats = {}
        for fatbin_path in fatbin_files:
            try:
                result = subprocess.run(["cuobjdump", "--dump-resource-usage", fatbin_path], capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                continue
            current_arch   = None
            current_kernel = None
            for line in result.stdout.splitlines():
                m_arch = re.search(r"arch = (\S+)", line)
                if m_arch:
                    current_arch   = m_arch.group(1)
                    current_kernel = None
                    continue
                m_func = re.search(r"Function (krnl_GPUTPC[^\s:]+)", line)
                if m_func:
                    if arch is None or current_arch == arch:
                        raw_name = m_func.group(1)
                        clean_name = (raw_name.replace("krnl_", "").replace("GPUTPC", "").rstrip(":"))
                        current_kernel = clean_name
                    else:
                        current_kernel = None
                    continue
                if current_kernel is not None:
                    m = re.search(r"REG:(\d+).*SHARED:(\d+)", line)
                    if m:
                        stats[current_kernel] = {"registers": int(m.group(1)), "shared_memory": int(m.group(2))}
                        current_kernel = None
        return stats

    @staticmethod
    def _detect_profiler(profiler):
        try:
            subprocess.run([profiler, "--version"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(f"{profiler} not found or not working properly.") from e
	
    @staticmethod
    def _run_and_log(cmd, global_log_path, run_log_path=None, timeout=None):
        run_fpath = run_log_path or "tmp_run.log"
        with open(run_fpath, "w") as run_f:
            try:
                result = subprocess.run(cmd, stdout=run_f, stderr=subprocess.STDOUT, timeout=timeout, check=True, text=True)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                run_f.write("ERROR: Benchmark stalled and timed out.\n")
                raise TimeoutError("Benchmark timed out")
            except subprocess.CalledProcessError as e:
                run_f.write(f"ERROR: Benchmark crashed. Return code: {e.returncode}\n")
                raise RuntimeError(f"Benchmark crashed with return code {e.returncode}")
        with open(global_log_path, "a") as global_f, open(run_fpath, "r") as run_f:
            shutil.copyfileobj(run_f, global_f)
        return returncode
            
    def rtc(self, dump=None, rtc_cache=True, return_stats=True, log_file=None):
        rtc_log = log_file or os.path.join(self.output_folder, "rtc.log")
        dump_path = dump or self.param_dump
        command = ["./ca", "--noEvents", "--sync", "-g", "--gpuType", self.gpu_lang, "--memSize", str(self.vRAM), "--RTCenable", "1", "--RTCTECHrunTest", "2", "--RTCTECHloadLaunchBoundsFromFile", dump_path]
        if rtc_cache:
            command += ["--RTCcacheOutput", "1"]
        if return_stats:
            command += ["--RTCTECHkeepTempFiles", "1"]
            for f in glob.glob("/tmp/o2cagpu_rtc_*"):
                os.remove(f)
        self._run_and_log(command, rtc_log, run_log_path=rtc_log, timeout=90)
        if return_stats:
            return self.get_rtc_HW_stats()

    def profile_benchmark(self, dataset=None, dump=None, RTC=True, run_log_file=None):
        self.dataset = dataset or self.dataset
        self.param_dump = dump or self.param_dump
        if RTC and self.backend == "nvidia":
            rtc_dump = ["./ca", "--noEvents", "--sync", "-g", "--gpuType", self.gpu_lang, "--memSize", str(self.vRAM), "--RTCenable", "1", "--RTCcacheOutput", "1", "--RTCTECHrunTest", "2", "--RTCTECHloadLaunchBoundsFromFile", self.param_dump]
        command = [self.profiler] + self.profiler_options
        command += ["./ca", "-e", self.dataset, "--sync", "-g", "--gpuType", self.gpu_lang, "--memSize", str(self.vRAM), "--preloadEvents"]
        if self.num_events and self.num_events > 0:
            command += ["-n", str(self.num_events)]
        else:
            command += ["--runs", str(self.num_runs)]
        if RTC:
            command += ["--RTCenable", "1", "--RTCcacheOutput", "1", "--RTCTECHloadLaunchBoundsFromFile", self.param_dump]
        if RTC and self.backend == "nvidia":
            self._run_and_log(rtc_dump, self.benchmark_backend_log, run_log_file, timeout=90)
        timeout = 60 * self.num_events if self.num_events and self.num_events > 1 else 60 * self.num_runs # stall timeout check
        self._run_and_log(command, self.benchmark_backend_log, run_log_file, timeout=timeout)
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
                with open(self.benchmark_backend_log, 'a') as f:
                    subprocess.run(command, shell=True, cwd=self.output_folder, stdout=f, stderr=f)

    def update_param_file(self, kernels_config, filename, modified_header_path = None, dump_path=None, log_file=None):
        tmp_file = f"/tmp/{os.path.basename(filename)}"
        shutil.copyfile(filename, tmp_file)
            
        macro_updates = {}
        kernel_updates = {}
        for key, value in kernels_config.items():
            if isinstance(value, dict):
                kernel_updates[key] = value
            elif key.startswith("PAR_"):
                macro_updates[f"GPUCA_{key}"] = value
        with open(tmp_file, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            for macro_name, macro_value in macro_updates.items():
                if stripped.startswith(f"#define {macro_name}"):
                    line = f"#define {macro_name} {macro_value}\n"
                    break
            for kernel_name, config in kernel_updates.items():
                define_name = f"GPUCA_LB_GPUTPC{kernel_name}"
                match = re.match(rf"^(\s*#define\s+{define_name}\s+)(.*)", line)
                if not match:
                    continue
                prefix = match.group(1)
                rest = match.group(2).strip()
                existing_vals = [v.strip() for v in rest.split(",") if v.strip()]
                if not existing_vals:
                    continue
                if "block_size" in config and config["block_size"] is not None:
                    if len(existing_vals) >= 1:
                        existing_vals[0] = str(config["block_size"])
                    else:
                        existing_vals = [str(config["block_size"])]
                if "blocks_per_sm" in config and config["blocks_per_sm"] is not None:
                    if len(existing_vals) >= 2:
                        existing_vals[1] = str(config["blocks_per_sm"])
                    else:
                        existing_vals.append(str(config["blocks_per_sm"]))
                new_values = ", ".join(existing_vals)
                line = f"{prefix}{new_values}\n"
                break
            new_lines.append(line)

        with open(tmp_file, "w") as f:
            f.writelines(new_lines)
        saved_copy = None
        if dump_path is not None:
            out_dir = os.path.dirname(os.path.abspath(dump_path))
            saved_copy = os.path.join(out_dir, os.path.basename(tmp_file))
            shutil.copyfile(tmp_file, saved_copy)
        if modified_header_path is not None:
            shutil.copyfile(tmp_file, modified_header_path)
        output = dump_path if dump_path else "parameters.out"
        root_command = (
            f"echo -e '#define PARAMETER_FILE \"{tmp_file}\"\\n"
            f"gInterpreter->AddIncludePath(\"'`pwd`'/include/GPU\");\\n"
            f".x share/GPU/tools/dumpGPUDefParam.C(\"{output}\")\\n.q\\n' | root -l -b"
        )
        if log_file:
            with open(log_file, "a") as f:
                subprocess.run(root_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(root_command, shell=True, check=True)

    def get_launch_bounds_from_param(self, kernel_name, filename):
        tmp_file = f"/tmp/{os.path.basename(filename)}"

        if not os.path.exists(tmp_file):
            raise FileNotFoundError(f"Temporary parameter file not found: {tmp_file}")

        define_name = f"GPUCA_LB_GPUTPC{kernel_name}"

        pattern = re.compile(rf"^\s*#define\s+{define_name}\s+(.*)")

        with open(tmp_file, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    values = [v.strip() for v in match.group(1).split(",") if v.strip()]
                
                    block_size = int(values[0]) if len(values) >= 1 else None
                    blocks_per_sm = int(values[1]) if len(values) >= 2 else None

                    return block_size, blocks_per_sm

        # Not found
        return None, None

    @staticmethod
    def _write_stats_to_csv(output_file, fieldnames, rows):
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            if not file_exists:
                writer.writerow(fieldnames)
            writer.writerows(rows)

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

    def compute_kernel_mean_time(self, kernel_name, block_size, grid_size, dataset=None, write_to_csv=True):
        dataset = dataset or self.dataset
        kernel_stats = self._get_df_from_raw()
        filtered = kernel_stats[kernel_stats["Kernel_Name"].str.contains(kernel_name)]
        if filtered.empty:
            return float('inf'), 0.0
        data = filtered["DurationMs"].to_numpy()
        mean = np.mean(data)
        std_dev = np.std(data)
        if write_to_csv:
            output_file = os.path.join(self.output_folder, f'{kernel_name}_stats.csv')
            row = [[block_size, grid_size, self.dataset, mean, std_dev]]
            BenchmarkBackend._write_stats_to_csv(output_file, ["block_size", "grid_size", "dataset", "mean", "std_dev"], row)
        return (mean, std_dev)

    def compute_step_mean_time(self, step_name, kernels_config, dataset=None, write_to_csv=True):
        dataset = dataset or self.dataset
        subviews = []
        durations = []
        self._get_views()
        for view in self.views:
            subview = view[view["Kernel_Name"].str.contains('|'.join(kernels_config.keys()))]
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
        if write_to_csv:
            output_file = os.path.join(self.output_folder, f'{step_name}_step_stats.csv')
            row = [[kernels_config, dataset, mean, std_dev]]
            BenchmarkBackend._write_stats_to_csv(output_file, ["kernels_config", "dataset", "mean", "std_dev"], row)
        return (mean, std_dev)
    
    def get_kernel_mean_time(self, kernel_name, block_size, grid_size, dataset=None, filename="defaultParams.h"):
        dataset = dataset or self.dataset
        kernels_config = {kernel_name: {"block_size": block_size, "grid_size": grid_size}}
        self.update_param_file(kernels_config, filename, log_file=self.benchmark_backend_log)
        try:
            self.profile_benchmark(dataset)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error during benchmark: {e}")
            return (float('inf'), 0.0)
        return self.compute_kernel_mean_time(kernel_name, block_size, grid_size, dataset)

    def get_step_mean_time(self, step_name, kernels_config, dataset=None, filename="defaultParams.h"):
        self.update_param_file(kernels_config, filename, log_file=self.benchmark_backend_log)
        dataset = dataset or self.dataset
        try:
            self.profile_benchmark(dataset)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error during step benchmark: {e}")
            return (float('inf'), 0.0)
        return self.compute_step_mean_time(step_name, kernels_config, dataset)

    def get_sync_mean_time(self, dump=None, dataset=None):
        dataset = dataset or self.dataset
        command = [
            "./ca", "-e", dataset, "--sync", "-g", "--gpuType", self.gpu_lang, "--memSize", str(self.vRAM), "--preloadEvents", "--RTCenable", "1"]
        if self.num_events and self.num_events > 0:
            command += ["-n", str(self.num_events)]
        else:
            command += ["--runs", str(self.num_runs)]
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

    def get_step_mean_time_no_RTC(self, step_name, kernels_config, arch, dataset=None, filename="src/GPU/GPUTracking/Definitions/Parameters/GPUParameters.csv"):
        try:
            self.update_csv_param_file(kernels_config, filename, arch, compile=True)
        except subprocess.CalledProcessError:
            print("[WARNING] Compilation failed")
            return (float('inf'), 0.0)
        dataset = dataset or self.dataset
        try:
            self.profile_benchmark(dataset, RTC=False)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error during step benchmark: {e}")
            return (float('inf'), 0.0)
        return self.compute_step_mean_time(step_name, kernels_config, dataset)
