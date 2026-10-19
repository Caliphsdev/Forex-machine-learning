# Working agreement for this project

This file tells Claude Code how to work here. The project is a modular research harness (see README.md
and docs/findings.md); new model types are added to it rather than started in a separate project.

## How we work together

- The user is learning by building. They type the code themselves; Claude shows what to write and
  which command to run, one step at a time, and the user pastes the output back.
- **Test first.** Show the test, let the user run it and see it fail (red), then show the code and
  let them see it pass (green). Ask for the `pytest` output; the passing count also reveals missed edits.
- **Verify before handing over.** Check new code and tests in a scratch copy outside the project first,
  so the user never debugs Claude's mistakes.
- Edit the user's files directly only when they ask ("fix", "change them for me", "add to the files").
  Otherwise show the code. When they do say fix, check the files rather than assuming what is missing.
- Explain the why in short notes (an "Insight" block is welcome), and keep replies focused.
- Commit and push only when asked. Never put credentials, account numbers or personal paths in the repo.

## How we do research

- **Baselines before models.** Always beat a simple baseline (majority guess, predict-zero, a linear
  model) before reaching for a neural network.
- **Write the pass mark before the experiment**, apply it literally, and report misses honestly,
  including when Claude's own prediction was wrong.
- **Develop on one period, confirm once on another.** Choose settings on a development set, run a single
  confirmation on held-out data, and never tune on test windows already seen. If a retry is needed,
  it needs new data.
- **No leakage, with tests that prove it.** Chronological splits with a purge gap equal to the label
  horizon; scalers and target standardisers fitted on training rows only; a test that rewrites the
  "future" and shows training cannot change.
- **Costs and realism belong in every backtest.** Use recorded costs, add a slippage stress test, and
  treat a positive backtest as an upper bound.
- Log every run (settings as JSON, per-fold metrics, summary) so results can be compared later.

## Code layout conventions

- `src/<package>/` with one layer per concern: ingest, db, features, labels, datasets, models,
  training, evaluation. Each layer talks only to the previous one, so pieces can be swapped.
- All SQL lives in one repository module; indicators are pure functions (Series in, Series out).
- Notebooks explore; `src/` decides. When something works in a notebook it moves into the package with a
  test. Use `%load_ext autoreload` / `%autoreload 2` at the top of notebooks.
- Chart functions return a figure and never call `plt.show()`. Keep one quiet, consistent style.
- Times are stored and compared in UTC.

## Environment (this machine)

- Windows 11, PowerShell, VS Code. Python **3.12** in `.venv` (3.14 is too new for the libraries).
- NVIDIA RTX 5060 Laptop GPU: install PyTorch with
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`. TensorFlow is avoided
  because it has no native Windows GPU support.
- Install the package in editable mode (`pip install -e .`); run the suite with `pytest`.
