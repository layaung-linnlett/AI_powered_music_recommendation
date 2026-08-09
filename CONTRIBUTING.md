# Contributing to Music Mood Classifier

This document covers how to set up the project locally, the code standards I
follow, and the process for submitting changes.

## Getting Started

### Prerequisites

- Python 3.12 or later (required by the pinned `numpy` and `scipy`)
- Git
- On macOS: `brew install libomp`, which LightGBM needs at import time

### Local Setup

```bash
# 1. Fork and clone the repository.
git clone https://github.com/layaung-linnlett/Music_Mood_Classifier.git
cd Music_Mood_Classifier

# 2. Create and activate a virtual environment.
python -m venv .venv
source .venv/bin/activate        # macOS and Linux
.venv\Scripts\activate           # Windows

# 3. Install all dependencies.
pip install -r requirements.txt

# 4. Place the raw dataset at data/raw/dataset.csv.
```

## Development Workflow

### Branches

All work should go on a feature branch, not directly on `main`. I use the
format `feature/<short-description>` or `fix/<short-description>`.

```bash
git checkout -b feature/add-new-feature
```

### Running the Pipeline

```bash
python -m src.data_loader       # verify dataset loads
python -m src.eda               # regenerate EDA figures and report
python -m src.preprocessing     # rebuild splits and pipeline
python -m src.model_training    # re-run CV, tuning, and training
python -m src.evaluation        # re-run test evaluation
```

### Running Tests

All tests must pass before submitting a pull request.

```bash
pytest tests/ -v
```

## Code Standards

### Style

- Follow PEP 8 with 4-space indentation.
- Maximum line length is 100 characters.
- All functions and classes need Google-style docstrings with type hints.

### Docstring Format

```python
def my_function(param: int) -> str:
    """One-line summary sentence.

    Longer description if needed.

    Args:
        param: Description of the parameter.

    Returns:
        Description of the return value.

    Raises:
        ValueError: If param is negative.
    """
```

### Constants

Keep magic numbers out of the logic. Define all constants at the top of the
file so they are easy to find and change.

### Comments

Write comments in full sentences. Skip commenting on lines that already read
clearly from the code itself.

## Commit Messages

- Write in the imperative mood: "Add feature" not "Added feature".
- Keep the subject line under 72 characters.
- Reference issue numbers where relevant: `Fix confusion matrix label order (#42)`.

## Pull Requests

1. Make sure all tests pass locally.
2. Update or add docstrings for any functions you changed.
3. If you added new packages, update `requirements.txt` with pinned versions.
4. Open a pull request against `main` with a clear description of what changed
   and why.
5. Request a review from a maintainer.

## Reporting Bugs

Open an issue on GitHub with:
- A clear title describing the problem.
- Steps to reproduce it.
- What you expected vs. what actually happened.
- Your Python version and operating system.

## Licence

Contributions are licensed under the same licence as this project (MIT).
