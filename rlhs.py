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
        self.measured_times = []
        self.kernels_configs = []
        self.kernel_sample_dict = {}
        self.output_folder = RLHSSearch.create_unique_folder(f"rlhs_search_{datetime.now().strftime('%d-%m-%Y')}") if output_folder is None else output_folder
        self.log(f"Created output folder: {self.output_folder}")
        self.backend = BenchmarkBackend(self.output_folder)
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"{timestamp} - {message}"
        print(full_message, flush=True)
        with open(os.path.join(self.output_folder, "rlhs.log"), "a") as log_file:
            log_file.write(full_message + "\n")

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
        self.measured_times = []
        self.kernels_configs = []
        self.param_keys = []
        self.param_values = []
        self.kernel_sample_dict = {}
        for kernel, params in kernels_param_space.items():
            for param, val in params.items():
                self.param_keys.append((kernel, param))
                self.param_values.append(np.array(val))
                self.kernel_sample_dict[kernel] = {'grid_size': [], 'block_size': []}

    def _perform_lhs(self, num_samples):
        sampler = qmc.LatinHypercube(d=len(self.param_values))
        lhs_samples = sampler.random(n=num_samples)
        for i, (kernel, param) in enumerate(self.param_keys): # iterates over (kernel_name, grid/block_size)
            indices = np.round(lhs_samples[:, i] * (len(self.param_values[i]) - 1)).astype(int)
            self.kernel_sample_dict[kernel][param] = self.param_values[i][indices]  # put values of samples in correct array

    def print_progress(self, iteration, total, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        best_index = -1 if len(self.measured_times) == 0 else np.argmin(self.measured_times)
        best_value = -1 if len(self.measured_times) == 0 else self.measured_times[best_index]

        for kernel_name, params in self.kernel_sample_dict.items():
            grid_size = params['grid_size'][iteration if iteration < len(params['grid_size']) else -1]
            block_size = params['block_size'][iteration if iteration < len(params['block_size']) else -1]
            print(f"Evaluating {kernel_name} with block size = {block_size}, Grid Size = {grid_size}.")
        print(f"Best so far: {best_value:.2f} ms.")
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}\n')
        sys.stdout.flush()

        if iteration < total:
            sys.stdout.write("\033[F" * (len(self.kernel_sample_dict.keys()) + 2))
            sys.stdout.flush()

    def _sample_kernel(self, kernels_param_space, beamtype, IR, num_samples):
        self._init_sampling(kernels_param_space)
        self._perform_lhs(num_samples)
        kernel_name = list(kernels_param_space.keys())[0]
        self.log(f"Optimising kernel: {kernel_name} with {beamtype} collisions at {IR}Hz.")

        for i in range(num_samples):
            sample = (self.kernel_sample_dict[kernel_name]['grid_size'][i], self.kernel_sample_dict[kernel_name]['block_size'][i])
            self.print_progress(i, num_samples, prefix='Initial Sampling', suffix='Complete', length=50)
            if tuple(sample) in self.kernels_configs: # configuration already sampled
                continue
            value = self.backend.get_kernel_mean_time(kernel_name, sample[1], sample[0], beamtype, IR)[0]
            self.measured_times.append(value)
            self.kernels_configs.append(sample)
        
        self.print_progress(num_samples, num_samples, prefix='Initial Sampling', suffix='Complete', length=50)
        best_index = np.argmin(self.measured_times)
        best_sample = self.kernels_configs[best_index]
        best_block_size = best_sample[1]
        best_grid_size = best_sample[0]

        self.log("First Sampling Results:")
        for i, config in enumerate(self.kernels_configs):
            print(f"Sample {i+1}: Block Size: {config[1]}, Grid Size: {config[0]} -> Mean time: {self.measured_times[i]:.2f} ms")
        self.log(f"Best configuration from first sampling: Block Size = {best_block_size}, Grid Size = {best_grid_size}")
        self.log(f"Mean kernel time: {self.measured_times[best_index]:.2f} ms.")

        previous_best_value = self.measured_times[best_index]

        for iteration in range(5):
            start_idx = len(self.measured_times)
            max_block_size = max(kernels_param_space[kernel_name]["block_size"])
            max_grid_size = max(kernels_param_space[kernel_name]["grid_size"])
            refined_block_sizes = np.array([max(64, best_block_size - 64), best_block_size, min(max_block_size, best_block_size + 64)])
            step_size = max(5, 60 // (iteration + 1))
            refined_grid_sizes = np.arange(max(60, best_grid_size - 120), min(max_grid_size, best_grid_size + 120), step_size)
            
            self.param_values = [refined_grid_sizes, refined_block_sizes]
            self._perform_lhs(num_samples)

            for i in range(num_samples):
                sample = (self.kernel_sample_dict[kernel_name]['grid_size'][i], self.kernel_sample_dict[kernel_name]['block_size'][i])
                self.print_progress(i, num_samples, prefix=f'Refining Iteration {iteration + 1}', suffix='Complete', length=50)
                if sample in self.kernels_configs: # configuration already sampled
                    continue
                value = self.backend.get_kernel_mean_time(kernel_name, sample[1], sample[0], beamtype, IR)[0]
                self.measured_times.append(value)
                self.kernels_configs.append(tuple(sample))

            self.print_progress(num_samples, num_samples, prefix=f'Refining Iteration {iteration + 1}', suffix='Complete', length=50)
            new_values = self.measured_times[start_idx:]
            if len(new_values) == 0:
                self.log(f"Iteration {iteration + 1} produced no new samples.")
                break
            new_configs = self.kernels_configs[start_idx:]
            refined_best_index = np.argmin(new_values)
            refined_best_value = new_values[refined_best_index]
            refined_best_sample = new_configs[refined_best_index]
            refined_best_block_size = refined_best_sample[1]
            refined_best_grid_size = refined_best_sample[0]
            
            self.log(f"Second Sampling Results (Refined) - Iteration {iteration + 1}:")
            for i, (config, value) in enumerate(zip(new_configs, new_values)):
                print(f"Sample {i+1}: Block Size: {config[1]}, Grid Size: {config[0]} -> Mean time: {value:.2f} ms")
            self.log(f"Best configuration from iteration {iteration + 1}: Block Size = {refined_best_block_size}, Grid Size = {refined_best_grid_size}")
            self.log(f"Mean kernel time: {new_values[refined_best_index]:.2f} ms.")

            if refined_best_value >= previous_best_value:
                self.log("Stopping refinement: new best configuration is not better than previous best.")
                break

            improvement = (previous_best_value - refined_best_value) / previous_best_value
            threshold = 0.02
            if improvement < threshold:
                self.log(f"Stopping refinement: improvement is less than {int(threshold*100)}%.")
                break

            previous_best_value = refined_best_value
            best_block_size = refined_best_sample[1]
            best_grid_size = refined_best_sample[0]

        return self.kernels_configs, self.measured_times

    def _sample_step(self, kernels_param_space, beamtype, IR, num_samples, step_name):
        self._init_sampling(kernels_param_space)
        self._perform_lhs(num_samples)
        self.log(f"Optimising step: {step_name} with {beamtype} collisions at {IR}Hz.")

        for i in range(num_samples):
            kernels_config = {}
            for kernel_name in self.kernel_sample_dict.keys():
                kernels_config[kernel_name] = {}
                kernels_config[kernel_name]['grid_size'] = self.kernel_sample_dict[kernel_name]['grid_size'][i]
                kernels_config[kernel_name]['block_size'] = self.kernel_sample_dict[kernel_name]['block_size'][i]
            self.print_progress(i, num_samples, prefix='Initial Sampling', suffix='Complete', length=50)
            if kernels_config in self.kernels_configs: # configuration already sampled
                continue
            mean_time, _ = self.backend.get_step_mean_time(step_name, kernels_config, beamtype, IR) # retain only mean, toss std_dev
            self.measured_times.append(mean_time)
            self.kernels_configs.append(kernels_config)
        
        self.print_progress(num_samples, num_samples, prefix='Initial Sampling', suffix='Complete', length=50)
        best_index = np.argmin(self.measured_times)

        self.log("First Sampling Results:")
        for i, config in enumerate(self.kernels_configs):
            print(f"Sample {i+1}: Mean time: {self.measured_times[i]:.2f} ms")
            for kernel_name in config.keys():
                print(f"Kernel: {kernel_name}, Block Size: {config[kernel_name]['block_size']}, Grid Size: {config[kernel_name]['grid_size']}")
                        
        self.log(f"Best mean step time: {self.measured_times[best_index]:.2f} ms.")
        for kernel_name in self.kernels_configs[best_index].keys():
            grid_size = self.kernels_configs[best_index][kernel_name]['grid_size']
            block_size = self.kernels_configs[best_index][kernel_name]['block_size']
            print(f"Kernel: {kernel_name}, Block Size: {block_size}, Grid Size: {grid_size}")

        previous_best_value = self.measured_times[best_index]

        for iteration in range(5):
            start_idx = len(self.measured_times)
            self.param_values = []
            for kernel_name in kernels_param_space.keys():
                max_block_size = max(kernels_param_space[kernel_name]["block_size"])
                max_grid_size = max(kernels_param_space[kernel_name]["grid_size"])
                best_block_size = self.kernel_sample_dict[kernel_name]['block_size'][best_index]
                best_grid_size = self.kernel_sample_dict[kernel_name]['grid_size'][best_index]
                refined_block_sizes = np.array([max(64, best_block_size - 64), best_block_size, min(max_block_size, best_block_size + 64)])
                step_size = max(5, 60 // (iteration + 1))
                refined_grid_sizes = np.arange(max(60, best_grid_size - 120), min(max_grid_size, best_grid_size + 120), step_size)
                self.param_values.append(np.array(refined_grid_sizes))
                self.param_values.append(np.array(refined_block_sizes))

            self._perform_lhs(num_samples)

            for i in range(num_samples):
                kernels_config = {}
                for kernel_name in self.kernel_sample_dict.keys():
                    kernels_config[kernel_name] = {}
                    kernels_config[kernel_name]['grid_size'] = self.kernel_sample_dict[kernel_name]['grid_size'][i]
                    kernels_config[kernel_name]['block_size'] = self.kernel_sample_dict[kernel_name]['block_size'][i]
                self.print_progress(i, num_samples, prefix=f'Refining Iteration {iteration + 1}', suffix='Complete', length=50)
                if kernels_config in self.kernels_configs: # configuration already sampled
                    continue
                mean_time, _ = self.backend.get_step_mean_time(step_name, kernels_config, beamtype, IR) # retain only mean, toss std_dev
                self.measured_times.append(mean_time)
                self.kernels_configs.append(kernels_config)

            self.print_progress(num_samples, num_samples, prefix=f'Refining Iteration {iteration + 1}', suffix='Complete', length=50)
            new_values = self.measured_times[start_idx:]
            if len(new_values) == 0:
                self.log(f"Iteration {iteration + 1} produced no new samples.")
                break
            new_configs = self.kernels_configs[start_idx:]
            refined_best_index = np.argmin(new_values)
            refined_best_value = new_values[refined_best_index]
            
            self.log(f"Second Sampling Results (Refined) - Iteration {iteration + 1}:")
            for i, config in enumerate(new_configs):
                print(f"Sample {i+1}: Mean time: {new_values[i]:.2f} ms")
                for kernel_name in config.keys():
                    print(f"Kernel: {kernel_name}, Block Size: {config[kernel_name]['block_size']}, Grid Size: {config[kernel_name]['grid_size']}")
            
            self.log(f"Best mean step time: {refined_best_value:.2f} ms.")
            for kernel_name in new_configs[refined_best_index].keys():
                grid_size = new_configs[refined_best_index][kernel_name]['grid_size']
                block_size = new_configs[refined_best_index][kernel_name]['block_size']
                print(f"Kernel: {kernel_name}, Block Size: {block_size}, Grid Size: {grid_size}")

            if refined_best_value >= previous_best_value:
                self.log("Stopping refinement: new best configuration is not better than previous best.")
                break

            improvement = (previous_best_value - refined_best_value) / previous_best_value
            threshold = 0.02
            if improvement < threshold:
                self.log(f"Stopping refinement: improvement is less than {int(threshold*100)}%.")
                break

            previous_best_value = refined_best_value

        return self.kernels_configs, self.measured_times

    def sample(self, kernels_param_space, beamtype, IR, num_samples=20, step_name="multi_kernel"):
        if len(kernels_param_space) == 1:
            return self._sample_kernel(kernels_param_space, beamtype, IR, num_samples)
        else:
            return self._sample_step(kernels_param_space, beamtype, IR, num_samples, step_name)

def main():
    configurations = [("pp", "100k"), ("pp", "2M"), ("pbpb", "5k"), ("pbpb", "50k")]
    configurations = [("pbpb", "5k")]
    sampler = RLHSSearch()

    kernels_param_space = {
        "TrackletConstructor": {
            "grid_size": 720,
            "block_size": 512
        },
        "TrackletSelector": {
            "grid_size": 360,
            "block_size": 128
        },
        "NeighboursCleaner": {
            "grid_size": 240,
            "block_size": 448
        },
        "NeighboursFinder": {
            "grid_size": 720,
            "block_size": 512
        },
        "StartHitsSorter": {
            "grid_size": 60,
            "block_size": 256
        },
        "CreateTrackingData": {
            "grid_size": 540,
            "block_size": 192
        },
        "StartHitsFinder": {
            "grid_size": 600,
            "block_size": 320
        },
        "ExtrapolationTracking": {
            "grid_size": 660,
            "block_size": 448
        },
    }

    for conf in configurations:
        beamtype, ir = conf
        mean, std_dev = sampler.backend.get_step_mean_time("multi_kernel", kernels_param_space, beamtype, ir)
        print(f"Step default mean time for {beamtype} at {ir}Hz: {mean:.2f} ms ± {std_dev:.2f} ms")
    return

    kernels_param_space = {
        "CompressionKernels_step0attached": {
            "grid_size": 840,
            "block_size": 64
        },
        "GMMergerFollowLoopers": {
            "grid_size": 120,
            "block_size": 256
        },
    }
    for conf in configurations:
        beamtype, ir = conf
        mean, std_dev = sampler.backend.get_step_mean_time("multi_kernel", kernels_param_space, beamtype, ir)
        print(f"Step default mean time for {beamtype} at {ir}Hz_1: {mean:.2f} ms ± {std_dev:.2f} ms")
    
    return

    kernels_param_space = {
        "CompressionKernels_step0attached": {
            "grid_size": 120,
            "block_size": 192
        },
        "GMMergerFollowLoopers": {
            "grid_size": 720,
            "block_size": 128
        },
    }
    for conf in configurations:
        beamtype, ir = conf
        mean, std_dev = sampler.backend.get_step_mean_time("multi_kernel", kernels_param_space, beamtype, ir)
        print(f"Step default mean time for {beamtype} at {ir}Hz: {mean:.2f} ms ± {std_dev:.2f} ms")
    
    return

    kernels_param_space = {
        "TrackletConstructor": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
        "TrackletSelector": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
        "NeighboursCleaner": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
        "NeighboursFinder": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
        "StartHitsSorter": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
        "CreateTrackingData": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
        "StartHitsFinder": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
        "ExtrapolationTracking": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 1025, 64)
        },
    }

    for conf in configurations:
        beamtype, ir = conf
        sampler.sample(kernels_param_space, beamtype=beamtype, IR=ir, num_samples=160, step_name="tracklet")

    return

    kernels_param_space = {
        "CompressionKernels_step0attached": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.array([64, 128, 192])
        },
        "GMMergerFollowLoopers": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 257, 64)
        },
    }

    for conf in configurations:
        beamtype, ir = conf
        sampler.sample(kernels_param_space, beamtype=beamtype, IR=ir, num_samples=40)

    return

    kernels_param_space = {
        "CompressionKernels_step0attached": {
            "grid_size": 720,
            "block_size": 192
        },
        "GMMergerFollowLoopers": {
            "grid_size": 600,
            "block_size": 64
        },
    }
    for conf in configurations:
        beamtype, ir = conf
        mean, std_dev = sampler.backend.get_step_mean_time("multi_kernel", kernels_param_space, beamtype, ir)
        print(f"Step default mean time for {beamtype} at {ir}Hz: {mean:.2f} ms ± {std_dev:.2f} ms")
    return

    kernels_param_space = {
        "GMMergerTrackFit": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 257, 64)
        },
    }

    for conf in configurations:
        beamtype, ir = conf
        sampler.sample(kernels_param_space, beamtype=beamtype, IR=ir, num_samples=20)

    return

    kernels_param_space = {
        "GMMergerSectorRefit": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 257, 64)
        },
    }

    for conf in configurations:
        beamtype, ir = conf
        sampler.sample(kernels_param_space, beamtype=beamtype, IR=ir, num_samples=20)

    kernels_param_space = {
        "GMMergerCollect": {
            "grid_size": np.arange(60, 901, 60),
            "block_size": np.arange(64, 513, 64)
        },
    }

    for conf in configurations:
        beamtype, ir = conf
        sampler.sample(kernels_param_space, beamtype=beamtype, IR=ir, num_samples=20)

    return

if __name__ == "__main__":
    main()