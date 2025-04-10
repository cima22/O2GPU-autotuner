import os
import subprocess
import csv
import numpy as np

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
            "--syncAsync", "-g", "--memSize", "30000000000", 
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

    def update_param_file(self, kernel_name, block_size, grid_size, filename="include/testParam.h"):
        sed_command = (
            f"sed -i '/^\\s*#define GPUCA_LB_GPUTPC{kernel_name} \\s*[0-9][0-9]*,\\? *[0-9]*/"
            f"{{s/\\s[0-9][0-9]*,\\? *[0-9]*/ {block_size}, {grid_size//60}/}}' {filename}"
        )
        os.system(sed_command)
        root_command = "root -l -q -b src/GPU/GPUTracking/Standalone/tools/dumpGPUDefParam.C"
        log_file = os.path.join(self.output_folder, 'time_kernels.log')
        with open(log_file, 'a') as f:
            subprocess.run(root_command, shell=True, stdout=f, stderr=subprocess.STDOUT)

    def _compute_durations(self, search_string):
        # Open the input CSV file
        input_file = f'{self.output_folder}/times_raw.csv'
        output_file = f'{self.output_folder}/{search_string}.csv'
        with open(input_file, 'r') as infile:
            reader = csv.DictReader(infile)
            # Prepare data for the output
            results = []
            for row in reader:
                kernel_name = row["Kernel_Name"]
                # Filter kernel names containing the search string
                if search_string in kernel_name:
                    start_time = int(row["Start_Timestamp"])
                    stop_time = int(row["Stop_Timestamp"])
                    # Calculate the duration in microseconds
                    duration = (stop_time - start_time) / 1000000.0
                    results.append({"Kernel_Name": kernel_name, "Duration (ms)": duration})
        
        # Write the results to the output CSV
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
        labels = [("sync", 1), ("sync", 2), ("async", 1), ("async", 2)]
        results = []
        stats = []
        for i, (proc_type, merger_idx) in enumerate(labels):
            subset = data[i::4]
            mean = np.mean(subset)
            std_dev = np.std(subset)
            row = [block_size, grid_size, beamtype, IR, proc_type, merger_idx, mean, std_dev]
            stats.append((mean, std_dev))
            results.append(row)
        if write_to_csv:
            self._write_stats_to_csv(output_file, ["block_size", "grid_size", "beamtype", "IR", "proc_type", "merger_idx", "mean", "std_dev"], results)
        return stats[0]

    def _compute_mean_times(self, kernel_name, block_size, grid_size, beamtype, IR):
        # Check which kernel to process
        if "GMMergerCollect" == kernel_name:
            return self._compute_MergerCollect_mean(block_size, grid_size, beamtype, IR)
        elif "GMMergerTrackFit" == kernel_name:
            return self._compute_MergerTrackFit_mean(block_size, grid_size, beamtype, IR)
        else:
            return self._compute_kernel_mean(kernel_name, block_size, grid_size, beamtype, IR)

    def get_kernel_mean_time(self, kernel_name, block_size, grid_size, beamtype, IR):
        self.update_param_file(kernel_name, block_size, grid_size)
        self.profile_benchmark(beamtype, IR)
        self._compute_durations(kernel_name)        
        return self._compute_mean_times(kernel_name, block_size, grid_size, beamtype, IR)
