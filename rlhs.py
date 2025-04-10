import numpy as np
from scipy.stats import qmc
from datetime import datetime
import os
import sys
from benchmark_backend import BenchmarkBackend

class RLHSSearch:

    def __init__(self, output_folder=None):
        self.param_keys = []
        self.param_values = []
        self.kernel_times = []
        self.kernel_configs = []
        self.output_folder = RLHSSearch.create_unique_folder(f"rlhs_search_{datetime.now().strftime('%d-%m-%Y')}") if output_folder is None else output_folder
        RLHSSearch.log(f"Created output folder: {self.output_folder}")
        self.backend = BenchmarkBackend(self.output_folder)

    @staticmethod
    def log(message):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    @staticmethod
    def create_unique_folder(base_name):
        folder_name = base_name
        index = 1
        while os.path.exists(folder_name):
            folder_name = f"{base_name}_{index}"
            index += 1
        os.makedirs(folder_name)
        return folder_name

    def _init_sampling(self, kernels_param_space):
        self.kernel_times = []
        self.kernel_configs = []
        self.param_keys = []
        self.param_values = []
        for kernel, params in kernels_param_space.items():
            for param, val in params.items():
                self.param_keys.append((kernel, param))
                self.param_values.append(val)

    def perform_lhs(self, num_samples):
        sampler = qmc.LatinHypercube(d=len(self.param_values))
        lhs_samples = sampler.random(n=num_samples)
        scaled_samples = np.zeros((num_samples, len(self.param_values)), dtype=int)
        for i, values in enumerate(self.param_values):
            indices = np.round(lhs_samples[:, i] * (len(values) - 1)).astype(int)
            scaled_samples[:, i] = values[indices]
        return scaled_samples

    def print_progress(self, iteration, total, current_sample, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        best_index = -1 if len(self.kernel_times) == 0 else np.argmin(self.kernel_times)
        best_sample = (-1, -1) if len(self.kernel_times) == 0 else self.kernel_configs[best_index]
        best_value = -1 if len(self.kernel_times) == 0 else self.kernel_times[best_index]

        print(f"Evaluating: Block Size = {current_sample[1]}, Grid Size = {current_sample[0]}")
        print(f"Best so far: Block Size = {best_sample[1]}, Grid Size = {best_sample[0]}, Value = {best_value:.2f}")
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}\n')
        sys.stdout.flush()

        if iteration < total:
            sys.stdout.write("\033[F" * 3)
            sys.stdout.flush()

    def sample(self, kernels_param_space, beamtype, IR, num_samples=20):
        self._init_sampling(kernels_param_space)
        scaled_samples = self.perform_lhs(num_samples)
        RLHSSearch.log(f"\nOptimising kernel: {list(kernels_param_space.keys())[0]} with {beamtype} collisions at {IR}Hz.")
        for i, sample in enumerate(scaled_samples):
            config_dict = {self.param_keys[j]: sample[j] for j in range(len(self.param_keys))}
            kernel_name = list(kernels_param_space.keys())[0]
            block_size = config_dict[(kernel_name, "block_size")]
            grid_size = config_dict[(kernel_name, "grid_size")]
            self.print_progress(i, num_samples, sample, prefix='Initial Sampling', suffix='Complete', length=50)
            if tuple(sample) in self.kernel_configs:
                continue
            value = self.backend.get_kernel_mean_time(kernel_name, block_size, grid_size, beamtype, IR)[0]
            self.kernel_times.append(value)
            self.kernel_configs.append(tuple(sample))
        
        self.print_progress(num_samples, num_samples, sample, prefix='Initial Sampling', suffix='Complete', length=50)
        best_index = np.argmin(self.kernel_times)
        best_sample = self.kernel_configs[best_index]
        best_block_size = best_sample[1]
        best_grid_size = best_sample[0]

        RLHSSearch.log("First Sampling Results:")
        for i, config in enumerate(self.kernel_configs):
            print(f"Sample {i+1}: Block Size: {config[1]}, Grid Size: {config[0]} -> Mean time: {self.kernel_times[i]:.2f} ms")
        RLHSSearch.log(f"Best configuration from first sampling: Block Size = {best_block_size}, Grid Size = {best_grid_size}")
        RLHSSearch.log(f"Mean kernel time: {self.kernel_times[best_index]:.2f} ms.")

        previous_best_value = self.kernel_times[best_index]
        start_idx = len(self.kernel_times)

        for iteration in range(5):
            max_block_size = max(kernels_param_space[kernel_name]["block_size"])
            max_grid_size = max(kernels_param_space[kernel_name]["grid_size"])
            refined_block_sizes = np.array([max(64, best_block_size - 64), best_block_size, min(max_block_size, best_block_size + 64)])
            step_size = max(5, 60 // (iteration + 1))
            refined_grid_sizes = np.arange(max(60, best_grid_size - 120), min(max_grid_size, best_grid_size + 120), step_size)
            
            self.param_values = [refined_grid_sizes, refined_block_sizes]
            refined_scaled_samples = self.perform_lhs(20)

            for i, sample in enumerate(refined_scaled_samples):
                block_size = sample[1]
                grid_size = sample[0]
                self.print_progress(i, num_samples, sample, prefix=f'Refining Iteration {iteration + 1}', suffix='Complete', length=50)
                if tuple(sample) in self.kernel_configs:
                    continue
                value = self.backend.get_kernel_mean_time(kernel_name, block_size, grid_size, beamtype, IR)[0]
                self.kernel_times.append(value)
                self.kernel_configs.append(tuple(sample))

            self.print_progress(num_samples, num_samples, sample, prefix=f'Refining Iteration {iteration + 1}', suffix='Complete', length=50)
            new_values = self.kernel_times[start_idx:]
            new_configs = self.kernel_configs[start_idx:]
            refined_best_index = np.argmin(new_values)
            refined_best_value = new_values[refined_best_index]
            refined_best_sample = new_configs[refined_best_index]
            refined_best_block_size = refined_best_sample[1]
            refined_best_grid_size = refined_best_sample[0]
            
            RLHSSearch.log(f"Second Sampling Results (Refined) - Iteration {iteration + 1}:")
            for i, (config, value) in enumerate(zip(new_configs, new_values)):
                print(f"Sample {i+1}: Block Size: {config[1]}, Grid Size: {config[0]} -> Mean time: {value:.2f} ms")
            RLHSSearch.log(f"Best configuration from iteration {iteration + 1}: Block Size = {refined_best_block_size}, Grid Size = {refined_best_grid_size}")
            RLHSSearch.log(f"Mean kernel time: {new_values[refined_best_index]:.2f} ms.")

            if refined_best_value >= previous_best_value:
                RLHSSearch.log("Stopping refinement: new best configuration is not better than previous best.")
                break

            improvement = (previous_best_value - refined_best_value) / previous_best_value
            threshold = 0.02
            if improvement < threshold:
                RLHSSearch.log(f"Stopping refinement: improvement is less than {int(threshold*100)}%.")
                break

            previous_best_value = refined_best_value
            best_block_size = refined_best_sample[1]
            best_grid_size = refined_best_sample[0]

        return scaled_samples, self.kernel_times

def main():
    configurations = [("pp", "100k"), ("pp", "2M"), ("pbpb", "5k"), ("pbpb", "50k")]
    sampler = RLHSSearch()
    
    kernels_param_space = {
        "CreateTrackingData": {
            "grid_size": 60,
            "block_size": 128
        },
        "StartHitsSorter": {
            "grid_size": 120,
            "block_size": 1024
        },
        "StartHitsFinder": {
            "grid_size": 60,
            "block_size": 1024
        },
        "TrackletConstructor": {
            "grid_size": 120,
            "block_size": 256
        },
        "TrackletSelector": {
            "grid_size": 480,
            "block_size": 256
        },
        "NeighboursFinder": {
            "grid_size": 60,
            "block_size": 1024
        },
        "NeighboursCleaner": {
            "grid_size": 60,
            "block_size": 896
        },
        "ExtrapolationTracking": {
            "grid_size": 60,
            "block_size": 256
        },
    }

    sampler.backend.get_step_mean_time("TrackletConstructor", kernels_param_space, "pbpb", "5k")

    return

    kernels_param_space = {
        "CompressionKernels_step1unattached": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.array([64, 128, 256, 512])
        },
    }
    for beamtype, IR in configurations:
        sampler.sample(kernels_param_space, beamtype, IR)

    return

    kernels_param_space = {
        "GMMergerCollect": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 513, 64)
        },
    }
    for beamtype, IR in configurations:
        sampler.sample(kernels_param_space, beamtype, IR)
    
    kernels_param_space = {
        "GMMergerTrackFit": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 257, 64)
        },
    }
    for beamtype, IR in configurations:
        sampler.sample(kernels_param_space, beamtype, IR)
    
    kernels_param_space = {
        "GMMergerSectorRefit": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 257, 64)
        },
    }
    for beamtype, IR in configurations:
        sampler.sample(kernels_param_space, beamtype, IR)

if __name__ == "__main__":
    main()