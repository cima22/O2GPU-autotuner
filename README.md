# O2GPU-autotuner
Automated tool to tune O2 GPU kernels - limited to TPC kernels.
## Installation
```bash
git clone https://github.com/cima22/O2GPU-autotuner.git
cd O2GPU-autotuner
pip install -e .
```
## Quickstart
Start tuning with
```bash
o2gpu-tune --dataset tpc_data_dump --output tuning_result_dir --time_budget 2h30m
```
Analyze the results with
```bash
o2gpu-analyze quick_tuning/
```
## Detailed guide
The tuner assumes that the repository has been cloned into the parent directory of the `standalone` directory that contains the `ca` executable. To personalize the execution, there are some environment variables that can be modified:
- `TUNER_WORKDIR` to set the directory containing the `ca` executable.
- `TUNER_PARAMETER_FILE` to set the parameter header file which will be used as base for the tuning. Default is `O2GPU_autotuner/defaults/defaultParamsNVIDIA.h`. For AMD cards, `O2GPU_autotuner/defaults/defaultParamsAMD.h` is present.
- `TUNER_DATASET` to set the tpc data dump to be used during tuning.

### Tuning
`o2gpu-tune` can accept the following arguments, which override the enviroment variables:
- `--output` to set the output directory.
- `--dataset` to set the tpc data dump.
- `--nEvents` to set the number of events to be used in the dataset.
- `--time_budget` to set the desired duration of the tuning. Can be set like: minutes (30m), hours (1h), or hh:mm (1:30).
- `--trials` to set the number of trials to use. Will override the trials computed by the time budget request.
- `--startup` to set the number of startup iterations to use. Will override the startup iterations computed by the time budget request.

### Analyzing
`o2gpu-analyze` analyzes the result of the tuning, writes an header with the optimised parameters and dumps a .par file. It requires the directory which has to be analyzed. It will automatically compare the performance of the tuned parameters using the settings used during tuning.
Example: `o2gpu-analyze tuning_dir` will output, at the end:
```
========== TIMING RESULTS ==========
[DEFAULT] mean = 4735.305000 ms | std = 0.000000 ms
[OPTIMIZED] mean = 4396.616000 ms | std = 0.000000 ms

Optimization SUCCESS
Gain      = 338.689000 ms
Speedup   = 1.077x
Improvement = 7.15%
====================================
```
### Tune spaces
The tuners optimises multiple ensemble of kernels (steps) at the same time. For a correct optimisation, these steps must be independent, i.e. they must not run at the same time on the GPU. The kernels within the same step can run concurrently. To specify a step, it is necesassary to create a directory containing one `yaml` file for each step. The default steps are present in `O2GPU_autotuner/tune_spaces`. To specify a new tune space directory, it is necessary to export the `TUNE_SPACE_DIR` env variable. \
The `block_size` parameter can be of two types:
- `type: values`: it expects also the field `values` that specifies the admissible values. Only multiples of warp size will be accepted. \
  Example where `CompressionKernels_step0attached` will be tested with these values for `block size` (on AMD GPUs, only multiples of 64 will be evaluated):
  ```
  CompressionKernels_step0attached:
    block_size:
      type: values
      values: [32, 64, 96, 128, 160, 192]
  ```
- `type: range`: it expects three variables:
  - `min`: the minimum multiple of the warpsize
  - `max_value`: the maximum value block size can have
  - `step`: step (in multiples of warp size) of the range \
  Example where `GMMergerFollowLoopers` will be evaluated with `block size` in a range from 32 to 256 for Nvidia, and from 64 to 256 for AMD GPUs (with step of 32 and 64, respectively):
  ```
  GMMergerFollowLoopers:
    block_size:
      type: range
      min: 1
      max_value: 256
      step: 1
  ```
