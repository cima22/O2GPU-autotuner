# test_backend.py

import os
import shutil
import pandas as pd
import re
from benchmark_backend.benchmarkBackend import BenchmarkBackend


def parse_lb(val):
    val = str(val).strip()

    if not (val.startswith("[") and val.endswith("]")):
        return None, None

    content = val[1:-1]
    parts = [p.strip() for p in content.split(",")]

    if len(parts) == 1:
        return parts[0], None

    elif len(parts) >= 2:
        block = parts[0]
        grid = ",".join(parts[1:]).strip()

        # try int conversion (optional)
        if isinstance(block, str) and block.isdigit():
            block = int(block)
        if isinstance(grid, str) and grid.isdigit():
            grid = int(grid)

        return block, grid

    return None, None


def build_kernels_config(flat_config):
    """
    Convert:
    CFClusterizer_grid → {"CFClusterizer": {"grid_size": ...}}
    CFClusterizer_block → {"CFClusterizer": {"block_size": ...}}
    """
    kernels = {}

    for key, value in flat_config.items():
        if key.endswith("_grid"):
            name = key.replace("_grid", "")
            kernels.setdefault(name, {})["grid_size"] = value

        elif key.endswith("_block"):
            name = key.replace("_block", "")
            kernels.setdefault(name, {})["block_size"] = value

    return kernels


def main():

    output_folder = "benchmark_output"
    os.makedirs(output_folder, exist_ok=True)

    csv_path = "src/GPU/GPUTracking/Definitions/Parameters/GPUParameters.csv"

    # --- Backup ---
    backup_path = csv_path + ".bak"
    shutil.copyfile(csv_path, backup_path)
    print(f"[INFO] Backup created: {backup_path}")

    try:
        backend = BenchmarkBackend(output_folder)

        # =========================
        # 🔥 YOUR CONFIG
        # =========================
        flat_config = {
            'CFClusterizer_grid': 416,
            'CFClusterizer_block': 768,
            'CFDecodeZSDenseLink_grid': 936,
            'CFNoiseSuppression_noiseSuppression_grid': 728,
            'CFDeconvolution_grid': 624,
            'CFPeakFinder_grid': 312,
            'CFCheckPadBaseline_grid': 520,
            'CFStreamCompaction_compactDigits_grid': 104,
            'CFNoiseSuppression_updatePeaks_grid': 520,
            'CFStreamCompaction_scanStart_grid': 104,
            'CFGather_grid': 520,
            'CFStreamCompaction_scanUp_grid': 624,
            'CFStreamCompaction_scanTop_grid': 520,
            'CFStreamCompaction_scanDown_grid': 832
        }

        kernels_config = build_kernels_config(flat_config)

        print("\n[DEBUG] Built kernels_config:")
        for k, v in kernels_config.items():
            print(f"  {k}: {v}")

        # =========================
        # Run update
        # =========================
        backend.get_step_mean_time_no_RTC(
            step_name="test",
            kernels_config=kernels_config,
            arch="MI100",
            dataset="o2-pbpb-10kHz-32"
        )

        df_after = pd.read_csv(csv_path)

        print("\n=== Checking updates ===")

        # =========================
        # 🔍 VALIDATION LOOP
        # =========================
        for kernel_name, params in kernels_config.items():

            # Match row strictly
            mask = df_after["Architecture"].astype(str).str.contains(
                kernel_name, case=True, na=False
            )

            if mask.sum() != 1:
                print(f"[WARNING] Skipping ambiguous kernel: {kernel_name}")
                continue

            row_name = df_after.loc[mask, "Architecture"].values[0]
            val_after = df_after.loc[mask, "MI100"].values[0]

            b_after, g_after = parse_lb(val_after)

            print(f"\nKernel: {row_name}")
            print(f"Value : {val_after}")
            print(f"Parsed: block={b_after}, grid={g_after}")

            # -------------------------
            # Assertions
            # -------------------------
            if "block_size" in params:
                assert str(b_after) == str(params["block_size"]), \
                    f"[FAIL] Block mismatch for {kernel_name}: got {b_after}, expected {params['block_size']}"

            if "grid_size" in params:
                expected_grid = params["grid_size"] // backend.nSMs

                assert str(expected_grid) in str(g_after), \
                    f"[FAIL] Grid mismatch for {kernel_name}: got {g_after}, expected {expected_grid}"

        print("\n✅ Full configuration test passed!")

    finally:
        # Restore original file
        #shutil.move(backup_path, csv_path)
        print(f"[INFO] CSV restored from backup")


if __name__ == "__main__":
    main()