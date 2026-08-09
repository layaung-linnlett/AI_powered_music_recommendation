"""Shared pytest configuration.

Most of this suite exercises the real pipeline rather than mocks, so it needs
two things that deliberately do not ship with the repository:

- the Spotify dataset in ``data/raw/`` (too large for git; see data/README.md)
- a trained model at ``models/final_model.pkl`` (a build artefact, gitignored)

When either is missing the affected tests are skipped with an explanatory
reason instead of failing, so a fresh clone reports a clean run. Once you have
placed the dataset and run ``python -m src.model_training``, the full suite
executes.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import DATA_RAW_DIR, MODELS_DIR  # noqa: E402

DATASET_PRESENT = DATA_RAW_DIR.is_dir() and any(DATA_RAW_DIR.glob("*.csv"))
MODEL_PRESENT = (MODELS_DIR / "final_model.pkl").is_file()

_DATASET_REASON = (
    f"needs the Spotify dataset in {DATA_RAW_DIR.relative_to(ROOT)}/ "
    "- see data/README.md"
)
_MODEL_REASON = (
    "needs a trained model at models/final_model.pkl "
    "- run: python -m src.model_training"
)

# Tests that read the dataset, keyed by module then class.
_NEEDS_DATASET = {
    "test_data_loader.py": None,  # every class in the module
    "test_preprocessing.py": {
        "TestPipelineOutput",
        "TestSplitRatios",
        "TestStratification",
    },
}

# Tests that load the trained model. test_label_encoder_file_exists is excluded
# because the label encoder is committed, so that check is meaningful without a
# training run.
_NEEDS_MODEL = {
    "test_model.py": {
        "TestModelLoading",
        "TestPredictShape",
        "TestPredictLabels",
        "TestPredictProba",
    },
}
_MODEL_EXEMPT = {"test_label_encoder_file_exists"}


def _matches(mapping: dict, item: pytest.Item) -> bool:
    classes = mapping.get(item.path.name)
    if item.path.name not in mapping:
        return False
    if classes is None:
        return True
    return item.cls is not None and item.cls.__name__ in classes


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip pipeline tests when their inputs are absent."""
    for item in items:
        if not DATASET_PRESENT and _matches(_NEEDS_DATASET, item):
            item.add_marker(pytest.mark.skip(reason=_DATASET_REASON))
            continue
        if (
            not MODEL_PRESENT
            and _matches(_NEEDS_MODEL, item)
            and item.name not in _MODEL_EXEMPT
        ):
            item.add_marker(pytest.mark.skip(reason=_MODEL_REASON))
