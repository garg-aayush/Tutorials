# Concepts

## Tabular Data

### Long vs Wide Format

**Long format** — one row per measurement. Columns are `(tag/id, timestamp, value)`. Every new tag or analyte adds rows, not columns. Sparse data stays compact.

**Wide format** — one row per timestamp. Every tag or analyte is its own column. Good for ML feature tables where you want all signals aligned on the same time index.

### How tables are stored in this project

| Table | Format | Why |
|---|---|---|
| `silver.timeseries.events` | Long | One row per tag per minute — 1,492 tags × 2.06B rows |
| `bronze.lims.measurements` | Long | 35 sampling points × varying analytes; matches historian convention |
| `features_10min_at_load.parquet` | Wide | ML input — pivoted from silver events; one row per 10-min interval, ~517 columns |

The pivot from long → wide happens in `build_training_dataset.py` at feature engineering time. When joining LIMS into the training parquet, you pivot only the analytes you need from `bronze.lims.measurements` the same way.
