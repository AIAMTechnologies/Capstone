# Data exports for modeling

The backend exposes a small utility to materialize the `leads` table so it can be used for offline modeling or analytics.

## Prerequisites

* Ensure the backend `.env` (or local environment) contains a valid `DATABASE_URL` pointing at the production-like PostgreSQL instance.
* Install the backend dependencies (minimum: `fastapi`, `psycopg2-binary`). For Parquet output you will also need `pandas` and `pyarrow` in your active environment.

## Exporting the dataset

Run the module directly from the repository root:

```bash
python -m backend.data_exports
```

This command will create a timestamped CSV export under `data/`.

### Include installer performance metrics

Add the `--include-installer-performance` flag to join the aggregated installer metrics (via `assigned_installer_id`).

```bash
python -m backend.data_exports --include-installer-performance
```

### Write Parquet files or custom paths

Specify the format explicitly to write Parquet (requires optional dependencies):

```bash
python -m backend.data_exports --format parquet
```

You can control the output location with `--output`. Relative paths will be created inside the `data/` folder, while absolute paths are respected:

```bash
python -m backend.data_exports --output leads_snapshot.csv
```

The script prints the final artifact path after a successful export so that the dataset can be easily located.
