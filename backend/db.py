import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
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


def get_db_connection():
    try:
        conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


def execute_query(query: str, params: tuple = None, fetch: bool = True):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                result = cursor.fetchall()
                return result
            else:
                conn.commit()
                return True
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        conn.close()
