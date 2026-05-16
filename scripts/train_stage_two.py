from __future__ import annotations

from utils.train import TrainingStage, run_training


def main() -> None:
    run_training(TrainingStage.STAGE_TWO)


if __name__ == "__main__":
    main()
