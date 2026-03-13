# test_backend.py

from benchmark_backend.benchmarkBackend import BenchmarkBackend

def main():

    # folder where profiler output will be written
    output_folder = "benchmark_output"

    backend = BenchmarkBackend(output_folder, debug=True)

    # example kernels to group as one "step"
    step_kernels = [
        "GPUTPCNeighboursFinder",
        "GPUTPCTrackletConstructor",
        "GPUTPCTrackletSelector"
    ]

    kernel = "CompressionKernels_step0"

    dataset = "o2-pbpb-5kHz-32"   # change to your dataset name if needed

    print("Running benchmark...")

    #mean, std = backend.get_step_mean_time(step_kernels=step_kernels, dataset=dataset, dump=None)
    mean, std = backend.get_kernel_mean_time(kernel, dataset=dataset, dump=None)

    print("\nStep timing results")
    print("-------------------")
    print("Mean time (ms):", mean)
    print("Std dev (ms):", std)


if __name__ == "__main__":
    main()
