"""Utility for exporting lead data to local artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from fastapi import HTTPException

from backend.main import execute_query


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def ensure_data_dir() -> Path:
    """Ensure that the data directory exists and return its Path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_installer_performance_columns() -> List[str]:
    """Return the ordered list of columns in the installer_performance table."""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'installer_performance'
        ORDER BY ordinal_position
    """
    rows = execute_query(query)
    return [row["column_name"] for row in rows]


def build_leads_query(include_installer_performance: bool) -> str:
    """Construct the SELECT statement for fetching leads (with optional join)."""
    select_columns = ["leads.*"]
    join_clause = ""

    if include_installer_performance:
        installer_columns = get_installer_performance_columns()
        if installer_columns:
            aliased_columns = [
                f'ip."{column}" AS installer_performance_{column}'
                for column in installer_columns
            ]
            select_columns.extend(aliased_columns)
        join_clause = " LEFT JOIN installer_performance ip ON ip.installer_id = leads.assigned_installer_id"

    select_clause = ", ".join(select_columns)
    query = f"SELECT {select_clause} FROM leads"
    if join_clause:
        query += join_clause
    query += " ORDER BY leads.id"
    return query


def fetch_leads(include_installer_performance: bool = False) -> List[Dict[str, object]]:
    """Fetch leads from the database with optional installer performance columns."""
    query = build_leads_query(include_installer_performance)
    try:
        rows = execute_query(query)
        return [dict(row) for row in rows]
    except HTTPException as exc:
        raise RuntimeError(f"Failed to fetch data: {exc.detail}") from exc


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> Path:
    """Write the provided rows to a CSV file and return the file path."""
    if not rows:
        output_path.write_text("")
        return output_path

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_parquet(rows: List[Dict[str, object]], output_path: Path) -> Path:
    """Write the provided rows to a Parquet file and return the file path."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Parquet export requires pandas and pyarrow to be installed."
        ) from exc

    dataframe = pd.DataFrame(rows)
    dataframe.to_parquet(output_path, index=False)
    return output_path


def export_leads(
    include_installer_performance: bool,
    output_format: str,
    output_filename: str | None = None,
) -> Path:
    """Fetch data and write it to disk in the specified format."""
    rows = fetch_leads(include_installer_performance)
    data_dir = ensure_data_dir()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    extension = "parquet" if output_format == "parquet" else "csv"

    if output_filename:
        output_path = Path(output_filename)
        if not output_path.is_absolute():
            output_path = data_dir / output_path
    else:
        output_path = data_dir / f"leads_export_{timestamp}.{extension}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        written_path = write_parquet(rows, output_path)
    else:
        written_path = write_csv(rows, output_path)

    return written_path


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export leads data to CSV or Parquet")
    parser.add_argument(
        "--include-installer-performance",
        action="store_true",
        help="Join installer_performance data using assigned installer id",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format (default: csv)",
    )
    parser.add_argument(
        "--output",
        help="Optional output filename. Relative paths are created under the data/ directory.",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> None:
    """Entry point for command line execution."""
    args = parse_args(arguments)
    output_path = export_leads(
        include_installer_performance=args.include_installer_performance,
        output_format=args.format,
        output_filename=args.output,
    )
    print(f"Exported leads to {output_path}")


if __name__ == "__main__":
    main()
