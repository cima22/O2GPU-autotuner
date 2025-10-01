import argparse
import os
import pickle
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Print best Optuna solution from a pickle file.")
    parser.add_argument("folder", type=str, help="Path to the folder containing the Optuna pickle file")
    args = parser.parse_args()

    # Build path to pickle file
    pkl_path = os.path.join(args.folder, "opt1", "indep_kernel_study.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: File not found at {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        study = pickle.load(f)

    print("=== Best Trial ===")
    print(f"Trial Number: {study.best_trial.number}")
    print(f"Value: {study.best_trial.value:.6f}")
    print("\n{:<34} {:>12} {:>12}".format("Kernel", "Block", "Grid"))
    print("-" * 60)

    params = study.best_trial.params
    grouped = defaultdict(dict)

    for key, value in params.items():
        if key.endswith("_grid"):
            prefix = key.replace("_grid", "")
            grouped[prefix]["grid"] = value
        elif key.endswith("_block"):
            prefix = key.replace("_block", "")
            grouped[prefix]["block"] = value
        else:
            grouped[key] = value  # Non-kernel parameters

    for key, val in grouped.items():
        if isinstance(val, dict):  # kernel with grid/block
            grid = val.get("grid", "-")
            block = val.get("block", "-")
            print(f"{key:<34} {str(block):>12} {str(grid):>12}")
        else:
            print(f"{key:34} {str(val):>25}")

if __name__ == "__main__":
    main()
