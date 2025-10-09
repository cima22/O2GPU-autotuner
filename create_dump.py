import os
import subprocess
import sys
import shutil
from O2GPU_autotuner.benchmark_backend.benchmarkBackend import BenchmarkBackend

def parse_read_results_output(output):
    parsed = {}

    lines = output.splitlines()[6:]  # Skip header lines
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        # Try to separate kernel name from values
        if parts[0].startswith("PAR_"):
            # Parameter-only line
            name = parts[0]
            value = int(parts[-1])
            parsed[name] = value
        else:
            # Kernel line, may have block and grid
            name = []
            values = []
            for p in parts:
                if p.isdigit() or p == "-":
                    values.append(p)
                else:
                    name.append(p)
            name = " ".join(name)
            entry = {}
            if len(values) >= 1 and values[0] != '-':
                entry['block_size'] = int(values[0])
            if len(values) >= 2 and values[1] != '-':
                entry['grid_size'] = int(values[1])
            if entry:
                parsed[name] = entry

    return parsed


def run_on_all_subfolders(folder):
    cumulative_results = {}

    for sub in sorted(os.listdir(folder)):
        sub_path = os.path.join(folder, sub)
        if not os.path.isdir(sub_path):
            continue

        try:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "read_results.py")
            result = subprocess.check_output(
                ['python3', script_path, sub_path],
                text=True,
                stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            print(f"Error in {sub_path}: {e.output}")
            continue

        parsed = parse_read_results_output(result)
        cumulative_results.update(parsed)

    return cumulative_results

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_dump.py <parent_directory>")
        exit(1)
    top_folder = sys.argv[1]

    results = run_on_all_subfolders(top_folder)

    for key, value in results.items():
        print(f"{key}: {value}")
    
    TUNER_WORKDIR = os.getenv("TUNER_WORKDIR", os.path.join(os.path.dirname(__file__), "../standalone"))
    try:
        os.chdir(TUNER_WORKDIR)
        beamtype = "pbpb"
        ir = "50k"
        backend = BenchmarkBackend("tmp")
        tmp_file = os.path.join("tmp", "defaultParamsH100.h")
        shutil.copy("defaultParamsH100.h", tmp_file)
        backend.update_param_file(results, filename=tmp_file)
        default_mean, std_dev = backend.get_sync_mean_time(beamtype=beamtype, IR=ir, dump="defaultH100.par")
        print(f"Sync mean time default for {beamtype} at {ir}Hz: {default_mean:.2f} ms ± {std_dev:.2f} ms")
        mean, std_dev = backend.get_sync_mean_time(beamtype=beamtype, IR=ir, dump="parameters.out")
        print(f"Sync mean time optimised for {beamtype} at {ir}Hz: {mean:.2f} ms ± {std_dev:.2f} ms")
        gain = 100.0 * (default_mean - mean) / default_mean
        print(f"Performance gain with optuna for {beamtype} at {ir}Hz: {gain:.2f}%")
        print()
    except Exception as e:
        print(f"[ERROR] Benchmark run failed: {e}")
        traceback.print_exc()
