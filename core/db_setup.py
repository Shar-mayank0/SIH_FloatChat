import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class DB:
    """Postgres helper for NeonDB with methods to init extensions and create tables.

    Usage:
        db = DB.from_env()  # reads NEON_CONN from .env
        db.init_db()        # creates extension and tables if not exist
    """

    def __init__(self, database_url: str, echo: bool = False) -> None:
        self.database_url = self._normalize_url(database_url)
        self.echo = echo
        self._engine: Optional[Engine] = None

    @staticmethod
    def _normalize_url(url: str) -> str:
        # Normalize postgres:// to postgresql:// for SQLAlchemy
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql://", 1)
        return url

    @classmethod
    def from_env(cls, env_var: str = "NEON_CONN", echo: bool = False) -> "DB":
        load_dotenv()
        conn_str = os.getenv(env_var)
        if not conn_str:
            raise RuntimeError(f"Set {env_var} in your .env")
        return cls(conn_str, echo=echo)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.database_url, echo=self.echo, pool_pre_ping=True)
        return self._engine

    def run_sql(self, sql: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql))

    def init_extensions(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    SCHEMA_SQL = """
    -- enable vector extension (Neon supports pgvector)
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Profiles table (one row per profile)
    CREATE TABLE IF NOT EXISTS argo_profiles (
        profile_id              INT PRIMARY KEY,
        platform_number         TEXT,
        project_name            TEXT,
        pi_name                 TEXT,
        cycle_number            INT,
        direction               TEXT,
        data_centre             TEXT,
        dc_reference            TEXT,
        data_state_indicator    TEXT,
        data_mode               TEXT,
        platform_type           TEXT,
        float_serial_no         TEXT,
        firmware_version        TEXT,
        wmo_instrument_type     TEXT,
        juld                    DOUBLE PRECISION,
        juld_qc                 TEXT,
        juld_location           DOUBLE PRECISION,
        latitude                DOUBLE PRECISION,
        longitude               DOUBLE PRECISION,
        position_qc             TEXT,
        positioning_system      TEXT,
        profile_pres_qc         TEXT,
        profile_temp_qc         TEXT,
        profile_psal_qc         TEXT,
        vertical_sampling_scheme TEXT,
        config_mission_number   INT,
        profile_date            DATE,
        embedding               vector(1536)
    );

    -- Measurements table (one row per depth level)
    CREATE TABLE IF NOT EXISTS measurements (
        id                      SERIAL PRIMARY KEY,
        profile_id              INT REFERENCES argo_profiles(profile_id) ON DELETE CASCADE,
        level                   INT,
        pres                    DOUBLE PRECISION,
        pres_adjusted           DOUBLE PRECISION,
        temp                    DOUBLE PRECISION,
        temp_adjusted           DOUBLE PRECISION,
        psal                    DOUBLE PRECISION,
        psal_adjusted           DOUBLE PRECISION,
        pres_adjusted_error     TEXT,
        temp_adjusted_error     TEXT,
        psal_adjusted_error     TEXT,
        pres_qc                 TEXT,
        pres_adjusted_qc        TEXT,
        temp_qc                 TEXT,
        temp_adjusted_qc        TEXT,
        psal_qc                 TEXT,
        psal_adjusted_qc        TEXT
    );
    """

    def create_schema(self) -> None:
        with self.engine.begin() as conn:
            for stmt in self.SCHEMA_SQL.split(";\n"):
                clean = stmt.strip()
                if clean:
                    conn.execute(text(clean))

    def init_db(self) -> None:
        self.init_extensions()
        self.create_schema()


# Module-level cached engine for reuse across the app
_ENGINE_SINGLETON: Optional[Engine] = None


def get_engine(echo: bool = False) -> Engine:
    """Return a singleton SQLAlchemy Engine built from NEON_CONN in .env.

    This creates the Engine on first call and reuses it afterward.
    """
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        db = DB.from_env(echo=echo)
        _ENGINE_SINGLETON = db.engine
    return _ENGINE_SINGLETON
