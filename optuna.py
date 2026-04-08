import argparse
import os
import optuna

from optimise import optimise  # reuse your function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--startup", type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    storage = f"sqlite:///{args.output}/study.db"

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.startup,
        constant_liar=False
    )

    study = optuna.create_study(
        study_name=os.getenv("TUNE_SPACE_NAME", "study"),
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True
    )

    study.optimize(optimise, n_trials=args.trials)

    print("\nBest result:")
    print(study.best_trial)


if __name__ == "__main__":
    main()