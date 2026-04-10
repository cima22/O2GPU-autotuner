#!/usr/bin/env python3

import argparse
import os
import optuna
from optuna.study import get_all_study_summaries


def load_best_from_db(db_path):
    storage = f"sqlite:///{db_path}"

    # Detect study name automatically
    summaries = get_all_study_summaries(storage=storage)
    if not summaries:
        raise RuntimeError("No studies found in DB")

    study_name = summaries[0].study_name

    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial

    return study_name, best.value, best.params


def main():
    parser = argparse.ArgumentParser(description="Extract best configs from Optuna DBs")
    parser.add_argument("folder", help="Folder containing .db files")
    args = parser.parse_args()

    folder = os.path.realpath(args.folder)

    if not os.path.isdir(folder):
        print(f"[ERROR] Not a directory: {folder}")
        return

    db_files = [f for f in os.listdir(folder) if f.endswith(".db")]
    if not db_files:
        print("[ERROR] No .db files found")
        return

    results = {}
    merged_config = {}

    print("\n========== EXTRACTING BEST CONFIGS ==========\n")

    for db in sorted(db_files):
        db_path = os.path.join(folder, db)
        kernel_name = os.path.splitext(db)[0]

        try:
            study_name, value, params = load_best_from_db(db_path)

            results[kernel_name] = {
                "value": value,
                "params": params
            }

            merged_config[kernel_name] = params

            print(f"{kernel_name}:")
            print(f"  study: {study_name}")
            print(f"  best value: {value}")
            print(f"  params: {params}\n")

        except Exception as e:
            print(f"[ERROR] Failed reading {db}: {e}")

    print("\n========== MERGED CONFIG ==========\n")
    print(merged_config)


if __name__ == "__main__":
    main()