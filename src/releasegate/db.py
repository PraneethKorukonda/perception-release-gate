"""DuckDB over the gate's output folder.

After `releasegate gate`, a report folder holds parquet files for both
models. DuckDB reads parquet in place, so "loading" is just creating views:

    objects   one row per box per model (model = 'current' / 'candidate')
    tracks    one row per object per model
    checks    the gate's verdicts
    images, videos

The queries in sql/ are the ones I actually used to find failure cases.
`releasegate sql <name>` runs one; `releasegate sql` lists them.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

VIEWS = ["objects", "tracks", "checks", "images", "videos"]


def connect(report_dir: str | Path) -> duckdb.DuckDBPyConnection:
    d = Path(report_dir)
    con = duckdb.connect()
    for name in VIEWS:
        path = d / f"{name}.parquet"
        if path.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    return con


def list_queries() -> dict[str, str]:
    out = {}
    for f in sorted(resources.files("releasegate").joinpath("sql").iterdir()):
        if f.name.endswith(".sql"):
            first = f.read_text().splitlines()[0].lstrip("- ").strip()
            out[f.name[:-4]] = first
    return out


def query_text(name: str) -> str:
    return resources.files("releasegate").joinpath("sql", f"{name}.sql").read_text()


def run(con: duckdb.DuckDBPyConnection, name_or_sql: str) -> pd.DataFrame:
    sql = query_text(name_or_sql) if name_or_sql in list_queries() else name_or_sql
    return con.execute(sql).df()
