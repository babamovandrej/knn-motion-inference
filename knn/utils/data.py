from pathlib import Path
from typing import Literal

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

type Split = Literal["train", "test"]

type Dataset = tuple[pd.DataFrame, "pd.Series[int]"]

ACTIVITY_LABELS: dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def load_split(
    split: Split,
    data_dir: Path = DATA_DIR,
) -> Dataset:
    X = pd.read_csv(
        data_dir / split / f"X_{split}.csv",
    )
    y = pd.read_csv(
        data_dir / split / f"y_{split}.csv",
    )["activity_id"].astype(int)

    if len(X) != len(y):
        raise ValueError(
            f"{split}: feature rows ({len(X)}) and labels ({len(y)}) differ."
        )

    return X, y


def load_groups(
    split: Split,
    data_dir: Path = DATA_DIR,
) -> "pd.Series[int] | None":
    path = data_dir / split / f"subjects_{split}.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)["subject_id"].astype(int)
