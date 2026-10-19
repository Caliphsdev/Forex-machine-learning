# Forex ML

A modular research project that asks one question honestly: **can an LSTM trained on MT5 price
history find a tradable edge in EURUSD?** Data flows MT5 -> SQLite -> features -> PyTorch (GPU),
and every result is judged with walk-forward validation, real recorded spreads and pass marks
written down *before* the experiment ran.

**Short answer (details in [docs/findings.md](docs/findings.md)):** direction has a tiny real edge
(about 51.5% accuracy) that does not survive retail trading costs; volatility, by contrast, is
clearly predictable (out-of-sample R2 of about 0.35-0.39) but did not turn the strategy
profitable. This is research code, not a trading system and not financial advice.

## Requirements

- Windows (the `MetaTrader5` package is Windows-only) with the MT5 terminal installed and logged in
  to a broker account. This project used an OctaFX demo account.
- Python **3.12** (3.14 is not supported by the dependencies).
- An NVIDIA GPU is optional but used when present. Developed on an RTX 5060 Laptop (Blackwell),
  which needs the CUDA 12.8 build of PyTorch.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build, installed separately
pip install -r requirements.txt                                        # or requirements.lock.txt for exact versions
pip install -e .                                                       # makes `import fxml` work everywhere
pytest                                                                 # 124 tests should pass
```

`requirements.lock.txt` is a full `pip freeze` of the environment this was built in (without torch,
which was `2.11.0+cu128`).

## Everyday commands

```powershell
python scripts/init_db.py        # create data/forex.db and its tables
python scripts/download_data.py  # pull bars from the open MT5 terminal (symbols/timeframes in config.py)
python scripts/train.py --name my-run --start 2015-01-01 --horizon 12     # walk-forward + costed backtest, logged to SQLite
python scripts/train.py --help   # all options
```

Every run is stored in the `runs` and `fold_metrics` tables of `data/forex.db`, with its full
settings as JSON. Read them back with `fxml.db.repository.load_runs` / `load_fold_metrics`.

## How it is organised

```text
src/fxml/
  config.py          paths, symbols, horizons and defaults in one place
  ingest/            MT5 connection and the downloader
  db/                SQLite schema, connection, and the ONLY place SQL lives (repository.py)
  utils/             broker-time -> UTC conversion
  features/          indicators (pure functions), direction features, volatility features
  labels/            what the model predicts: forward return, direction, forward volatility
  datasets/          modelling tables, walk-forward splits, scaler, PyTorch window dataset
  models/            two-headed LSTM, linear baseline, model registry
  training/          trainer, one-fold run, walk-forward runner, baselines, full pipeline
  evaluation/        metrics, cost-aware backtest, charts
scripts/             command-line entry points (init_db, download_data, train) and throwaway probes
notebooks/           01 data check, 02 feature diagnostics, 03 first fold, 04 horizon study,
                     05 confirmation run, 06 volatility, 07 volatility filter
tests/               124 tests, written before the code they cover
docs/findings.md     what was tried and what was found
```

Each layer only talks to the one before it, so any piece (the model, the features, the cost model)
can be swapped without touching the rest.

## Conventions worth keeping

- **Time is UTC everywhere.** The broker clock is New York time + 7 hours, converted once in
  `utils/timeutils.py`. Data before 2015 can have timestamps off by up to an hour and a placeholder
  spread column, so `TRAIN_START` is 2015-01-01.
- **No leakage, and tests that prove it.** Splits are chronological with a gap equal to the label
  horizon; scalers and target standardisers are fitted on training rows only; every fold has a test
  showing that rewriting the "future" cannot change training.
- **Judge by pre-registered bars.** Write the pass mark before running an experiment, apply it
  literally, and do not tune on test windows you have already looked at. The 2015-2026 windows have
  been examined many times; new decisions need new data (for example a forward test on bars that
  arrive after today).
- **Costs are part of the result.** The backtest charges each bar's recorded spread (about 0.76 bps
  per trade on average) plus an optional slippage stress.
- **Notebooks explore, `src/` decides.** When something works in a notebook it moves into the package
  with a test. Use `%load_ext autoreload` and `%autoreload 2` at the top of a notebook so edits to
  `src/` are picked up.
