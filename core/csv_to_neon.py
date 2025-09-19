import os
from typing import Iterable, List, Tuple, Optional
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from sqlalchemy.engine import Engine
from psycopg2.extras import execute_values

from .db_setup import DB


PROFILE_COL_MAP = {
    "profile_id": "profile_id",
    "platform_number": "platform_number",
    "project_name": "project_name",
    "pi_name": "pi_name",
    "cycle_number": "cycle_number",
    "direction": "direction",
    "data_centre": "data_centre",
    "dc_reference": "dc_reference",
    "data_state_indicator": "data_state_indicator",
    "data_mode": "data_mode",
    "platform_type": "platform_type",
    "float_serial_no": "float_serial_no",
    "firmware_version": "firmware_version",
    "wmo_inst_type": "wmo_instrument_type",
    "juld": "juld",
    "juld_qc": "juld_qc",
    "juld_location": "juld_location",
    "latitude": "latitude",
    "longitude": "longitude",
    "position_qc": "position_qc",
    "positioning_system": "positioning_system",
    "profile_pres_qc": "profile_pres_qc",
    "profile_temp_qc": "profile_temp_qc",
    "profile_psal_qc": "profile_psal_qc",
    "vertical_sampling_scheme": "vertical_sampling_scheme",
    "config_mission_number": "config_mission_number",
    "date": "profile_date",
}

MEAS_COLS: List[str] = [
    "profile_id",
    "level",
    "pres",
    "pres_adjusted",
    "temp",
    "temp_adjusted",
    "psal",
    "psal_adjusted",
    "pres_adjusted_error",
    "temp_adjusted_error",
    "psal_adjusted_error",
    "pres_qc",
    "pres_adjusted_qc",
    "temp_qc",
    "temp_adjusted_qc",
    "psal_qc",
    "psal_adjusted_qc",
]


def _normalize_object_series(s: pd.Series) -> pd.Series:
    return s.apply(
        lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else None if v in ("None", "nan") else v
    )


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = _normalize_object_series(df[col])
    return df


def _to_python_scalar(value):
    import numpy as np
    if value is None:
        return None
    # pandas NA
    if isinstance(value, (type(pd.NA),)):
        return None
    # pandas missing check
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    # numpy scalar
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    # pandas Timestamp/Timedelta
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, pd.Timedelta):
        return value.to_pytimedelta()
    # bytes
    if isinstance(value, (bytes, bytearray)):
        return value.decode('utf-8', errors='ignore')
    return value


def _df_to_python(df: pd.DataFrame) -> pd.DataFrame:
    # Convert NaNs to None, numpy scalars to native
    df = df.where(pd.notna(df), None)
    return df.map(_to_python_scalar)


def _extract_profile_id_from_filename(path: Path) -> Optional[int]:
    """Extract integer profile id from filename pattern ..._R########_....csv"""
    import re
    m = re.search(r"R(\d+)", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _prepare_profiles(df: pd.DataFrame, profile_id: Optional[int]) -> pd.DataFrame:
    # force profile_id from filename if available
    df = df.copy()
    if profile_id is not None:
        df["profile_id"] = profile_id
    cols_present = [c for c in PROFILE_COL_MAP.keys() if c in df.columns]
    if not cols_present:
        return pd.DataFrame(columns=list(PROFILE_COL_MAP.values()))
    subset = df[cols_present].drop_duplicates(subset=["profile_id"], keep="first").copy()
    subset = _normalize_dataframe(subset)
    # rename to schema names
    subset = subset.rename(columns=PROFILE_COL_MAP)
    # convert profile_date from string to date
    if "profile_date" in subset.columns and not subset["profile_date"].isna().all():
        subset["profile_date"] = pd.to_datetime(subset["profile_date"], errors="coerce").dt.date
    # make types psycopg2-friendly
    subset = _df_to_python(subset)
    return subset


def _prepare_measurements(df: pd.DataFrame, profile_id: Optional[int]) -> pd.DataFrame:
    # force profile_id from filename if available
    df = df.copy()
    if profile_id is not None:
        df["profile_id"] = profile_id
    cols_present = [c for c in MEAS_COLS if c in df.columns]
    if not cols_present:
        return pd.DataFrame(columns=MEAS_COLS)
    subset = df[cols_present].copy()
    subset = _normalize_dataframe(subset)
    # make types psycopg2-friendly
    subset = _df_to_python(subset)
    return subset


def _upsert_profiles(engine: Engine, rows: Iterable[Tuple]) -> None:
    if not rows:
        return
    cols_sql = (
        "profile_id, platform_number, project_name, pi_name, cycle_number, direction, "
        "data_centre, dc_reference, data_state_indicator, data_mode, platform_type, float_serial_no, firmware_version, "
        "wmo_instrument_type, juld, juld_qc, juld_location, latitude, longitude, position_qc, positioning_system, "
        "profile_pres_qc, profile_temp_qc, profile_psal_qc, vertical_sampling_scheme, config_mission_number, profile_date"
    )
    insert_sql = f"""
    INSERT INTO argo_profiles ({cols_sql})
    VALUES %s
    ON CONFLICT (profile_id) DO UPDATE SET
        platform_number = EXCLUDED.platform_number,
        project_name = EXCLUDED.project_name,
        pi_name = EXCLUDED.pi_name,
        cycle_number = EXCLUDED.cycle_number,
        direction = EXCLUDED.direction,
        data_centre = EXCLUDED.data_centre,
        dc_reference = EXCLUDED.dc_reference,
        data_state_indicator = EXCLUDED.data_state_indicator,
        data_mode = EXCLUDED.data_mode,
        platform_type = EXCLUDED.platform_type,
        float_serial_no = EXCLUDED.float_serial_no,
        firmware_version = EXCLUDED.firmware_version,
        wmo_instrument_type = EXCLUDED.wmo_instrument_type,
        juld = EXCLUDED.juld,
        juld_qc = EXCLUDED.juld_qc,
        juld_location = EXCLUDED.juld_location,
        latitude = EXCLUDED.latitude,
        longitude = EXCLUDED.longitude,
        position_qc = EXCLUDED.position_qc,
        positioning_system = EXCLUDED.positioning_system,
        profile_pres_qc = EXCLUDED.profile_pres_qc,
        profile_temp_qc = EXCLUDED.profile_temp_qc,
        profile_psal_qc = EXCLUDED.profile_psal_qc,
        vertical_sampling_scheme = EXCLUDED.vertical_sampling_scheme,
        config_mission_number = EXCLUDED.config_mission_number,
        profile_date = EXCLUDED.profile_date
    """
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        try:
            execute_values(cur, insert_sql, rows, page_size=10_000)
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def _insert_measurements(engine: Engine, rows: Iterable[Tuple]) -> None:
    if not rows:
        return
    cols_sql = (
        "profile_id, level, pres, pres_adjusted, temp, temp_adjusted, psal, psal_adjusted, "
        "pres_adjusted_error, temp_adjusted_error, psal_adjusted_error, pres_qc, pres_adjusted_qc, temp_qc, temp_adjusted_qc, psal_qc, psal_adjusted_qc"
    )
    insert_sql = f"INSERT INTO measurements ({cols_sql}) VALUES %s"
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        try:
            execute_values(cur, insert_sql, rows, page_size=20_000)
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def ingest_csv_folder_to_neon(csv_folder: str = "data/csv", chunk_rows: int = 200_000) -> None:
    console = Console()
    db = DB.from_env()
    engine = db.engine
    csv_dir = Path(csv_folder)
    if not csv_dir.exists():
        console.print(f"[red]CSV folder not found: {csv_dir}[/red]")
        return

    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        console.print(f"[yellow]No CSV files found in {csv_dir}[/yellow]")
        return

    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Ingesting CSVs into Neon", total=len(files))
        for csv_path in files:
            progress.update(task, description=f"Ingesting {csv_path.name}")
            try:
                profile_id = _extract_profile_id_from_filename(csv_path)
                for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
                    profiles_df = _prepare_profiles(chunk, profile_id)
                    if not profiles_df.empty:
                        rows = [
                            tuple(_to_python_scalar(profiles_df.get(col, pd.Series([None]*len(profiles_df))).iloc[i]) for col in [
                                "profile_id","platform_number","project_name","pi_name","cycle_number","direction",
                                "data_centre","dc_reference","data_state_indicator","data_mode","platform_type","float_serial_no","firmware_version",
                                "wmo_instrument_type","juld","juld_qc","juld_location","latitude","longitude","position_qc","positioning_system",
                                "profile_pres_qc","profile_temp_qc","profile_psal_qc","vertical_sampling_scheme","config_mission_number","profile_date"
                            ])
                            for i in range(len(profiles_df))
                        ]
                        _upsert_profiles(engine, rows)

                    meas_df = _prepare_measurements(chunk, profile_id)
                    if not meas_df.empty:
                        rows_m = [
                            tuple(_to_python_scalar(meas_df.get(col, pd.Series([None]*len(meas_df))).iloc[i]) for col in MEAS_COLS)
                            for i in range(len(meas_df))
                        ]
                        _insert_measurements(engine, rows_m)
                console.print(f"[green]✓[/green] {csv_path.name}")
            except Exception as e:
                console.print(f"[red]✗ Failed {csv_path.name}: {e}")
            progress.advance(task)

    console.print("[bold green]Ingestion complete[/bold green]")
