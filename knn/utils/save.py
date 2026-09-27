from pathlib import Path

import joblib

from .bundle import ModelBundle


def save_model(
    bundle: ModelBundle,
    path: Path,
) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        model_path,
    )
