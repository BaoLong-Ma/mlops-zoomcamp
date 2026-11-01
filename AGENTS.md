# AGENTS.md — MLOps Zoomcamp

This repo is the MLOps Zoomcamp course (DataTalks.Club). It is a **course content
repository**, not a software project. Each numbered directory (`01-intro/` through
`07-project/`) is a self-contained module with lesson Markdown files, Jupyter
notebooks, and supporting data. There is no build system, no test suite, and no
package boundary. Work is done per-module.

## Repo layout

- `01-intro/` … `07-project/` — one directory per course module. Each contains
  numbered lesson files (`01-*.md` … `NN-*.md`), a `module.yaml` (unit list),
  `meta.json` (video URLs), a `README.md`, and optionally notebooks/data.
- `cohorts/` — frozen, self-contained copies of past live cohorts (2022–2025).
  **Do not edit these.** They are historical archives.
- `cohorts/self-paced/` — represents the current self-paced delivery.
- `course.yaml` — course-level metadata (title, cohorts, description).
- `summary/` — per-module study summaries in French (user-generated).
- `generate/` — utility notebook for generating module pages (thumbnails, etc.).
- `research/` — internal strategy docs.
- `.github/workflows/` — CI/CD for `06-best-practices/code/` only (pytest, pylint,
  terraform plan). Not relevant to other modules.

## Module 2: Experiment Tracking (02-experiment-tracking/)

This is the most code-heavy module and the one with the most local state.

### Environment

- A `.venv/` (uv-managed, Python 3.12) already exists in this directory.
  Activate it with `source .venv/bin/activate` or use `.venv/bin/python` directly.
- Dependencies are in `requirements.txt`: mlflow, jupyter, scikit-learn, pandas,
  seaborn, hyperopt, xgboost, fastparquet, boto3.
- Installed versions (as of 2026-10): mlflow 3.17.2, xgboost 3.4.1, hyperopt 0.3.0,
  scikit-learn 1.9.1.

### MLflow local state (do not delete)

- `mlflow.db` — SQLite backend store. Contains 4 experiments, 24 runs, and the
  `nyc-taxi-regressor` registered model (versions 1–3, aliases: production,
  staging, challenger, bad).
- `mlruns/` — local artifact storage. `mlruns/2/` holds the `nyc-taxi-experiment`
  runs and their artifacts (model files, preprocessor, feature importance plots).
  `mlruns/2/models/` holds logged model artifacts (`m-*` directories).
- `models/` — pickled model files (`lin_reg.bin`, `lasso.bin`, `lasso_test1.bin`,
  `preprocessor.b`). These are the raw pickle outputs from the notebook.
- `preprocessor/` — downloaded preprocessor artifact (from MLflow run).
- `data/` — NYC Green Taxi parquet files (Jan–Mar 2021). `.gitignore` excludes
  `*.parquet` and `data/`, so these are local-only.

### Key notebooks

- `duration-prediction.ipynb` — main notebook. Walks through: data loading,
  feature engineering (DictVectorizer), LinearRegression baseline, Lasso,
  XGBoost hyperparameter search with Hyperopt + MLflow tracking, autologging,
  model logging, and model loading. The notebook has been run and contains
  outputs.
- `model-registry.ipynb` — demonstrates `MlflowClient`, `search_experiments`,
  `search_runs`, `register_model`, `transition_model_version_stage` (deprecated),
  `set_registered_model_alias`, and model comparison by alias.
- `running-mlflow-examples/scenario-{1,2,3}.ipynb` — three deployment scenarios
  (local file, local server + SQLite, remote AWS). Scenario 3 requires AWS
  credentials and is not runnable locally.

### Commands

```bash
# Start MLflow UI (local, SQLite backend — enables model registry)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start MLflow server (for scenario-2 style setup)
mlflow server --backend-store-uri sqlite:///mlflow.db
```

### MLflow API gotchas

- `mlflow.list_experiments()` was removed. Use `mlflow.search_experiments()` or
  `client.search_experiments()` instead.
- `client.transition_model_version_stage()` is **deprecated since MLflow 2.9**.
  Use `client.set_registered_model_alias()` and tags instead. The notebook
  still calls it (with a FutureWarning) for demonstration purposes.
- `mlflow.sklearn.log_model()` and `mlflow.xgboost.log_model()` use the `name=`
  parameter in recent MLflow versions. The `artifact_path=` parameter is
  deprecated (you'll see a warning).
- `reg:linear` objective in XGBoost is deprecated in favor of `reg:squarederror`.
  The notebook still uses `reg:linear` (you'll see a warning).
- When registering a model from a run, the `model_uri` format is
  `runs:/<run_id>/<model_name>`. If the run has no artifact at that path, MLflow
  falls back to the logged model ID (`models:/m-*`).

### Data

- `data/green_tripdata_2021-{01,02,03}.parquet` — NYC Green Taxi trip data.
  Columns use `lpep_` prefix (e.g. `lpep_pickup_datetime`, `lpep_dropoff_datetime`),
  unlike Yellow Taxi which uses `tpep_`.
- Target: `duration` in minutes, computed as
  `(dropoff - pickup).total_seconds() / 60`. Filter to 1–60 minutes.
- Features: `PU_DO` (concatenated pickup+dropoff location IDs as string) and
  `trip_distance`. Encoded with `DictVectorizer`.

## Homework (cohorts/2025/02-experiment-tracking/)

The 2025 homework uses **2023** Green Taxi data (not 2021). Scripts are in
`cohorts/2025/02-experiment-tracking/homework/`:

- `preprocess_data.py` — loads raw parquet, fits DictVectorizer, saves
  `dv.pkl`, `train.pkl`, `val.pkl`, `test.pkl` to `./output/`.
- `train.py` — trains RandomForestRegressor, computes RMSE. Needs MLflow
  autologging added.
- `hpo.py` — Hyperopt search over RandomForest params. Needs MLflow logging
  added to the `objective` function.
- `register_model.py` — selects top-N runs, evaluates on test set, registers
  best model. Needs `search_runs` + `register_model` calls completed.

The homework scripts expect a tracking server at `http://127.0.0.1:5000`
(scenario-2 style). Start one before running `hpo.py` or `register_model.py`.

## Conventions

- Lesson Markdown files use YAML frontmatter with `video_url`, `code` (list of
  label/path pairs), `prev_url`, `next_url`.
- `module.yaml` uses `schema_version: 2` with `content_id` (UUID) and `units`
  list. `meta.json` maps units to YouTube video URLs.
- `cohorts/` directories are frozen. Never modify them.
- The root `README.md` is the course landing page. Module `README.md` files are
  module-level navigation pages.
- Images are in each module's `images/` directory, referenced by relative path
  in the Markdown.
- `.gitignore` excludes: `data/`, `*.parquet`, `*.csv`, `*.db`, `*.bin`,
  `.venv/`, `**/artifacts/`, `**/models/`, `__pycache__/`, `**.env`.
