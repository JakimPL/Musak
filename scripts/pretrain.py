from scripts.utils.train import TrainingStage, run_training


def main() -> None:
    run_training(TrainingStage.PRETRAINING)


if __name__ == "__main__":
    main()
