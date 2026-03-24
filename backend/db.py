import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException

logger = logging.getLogger("lead_allocation")

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://q4gems_admin:890*()iopIOP@capstone25.postgres.database.azure.com:5432/capstone25db")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 480
    ML_PUBLIC_API_KEY = os.getenv("ML_PUBLIC_API_KEY")
    GEOCODING_API_KEY = os.getenv("GEOCODING_API_KEY", "")
    GEOCODING_PROVIDER = os.getenv("GEOCODING_PROVIDER", "nominatim")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


settings = Settings()

# -------------------------------------------------------
# Connection pool — keeps 2–10 live connections to Azure
# Postgres so every execute_query() gets a ready connection
# instead of paying the ~150 ms TCP handshake each time.
# -------------------------------------------------------
_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            _pool = pg_pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=30,
                dsn=settings.DATABASE_URL,
                cursor_factory=RealDictCursor,
            )
    return _pool


def get_db_connection():
    """Borrow a connection from the pool, retrying briefly on exhaustion."""
    pool = _get_pool()
    last_err: Exception = RuntimeError("pool unavailable")
    for attempt in range(10):
        try:
            return pool.getconn()
        except pg_pool.PoolError as e:
            last_err = e
            if attempt < 9:
                time.sleep(0.2 * (attempt + 1))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Database connection pool exhausted: {last_err}")


def release_db_connection(conn) -> None:
    """Return a connection to the pool."""
    try:
        _get_pool().putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def execute_query(query: str, params: tuple = None, fetch: bool = True):
    """Execute a query using a pooled connection (with retry on exhaustion)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            else:
                conn.commit()
                return True
    except HTTPException:
        raise
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        release_db_connection(conn)
