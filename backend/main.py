# ============================================
# LEAD ALLOCATION SYSTEM - FastAPI Backend
# ============================================

# requirements.txt:
# fastapi==0.104.1
# uvicorn[standard]==0.24.0
# psycopg2-binary==2.9.9
# pydantic==2.5.0
# pydantic-settings==2.1.0
# python-jose[cryptography]==3.3.0
# passlib[bcrypt]==1.7.4
# python-multipart==0.0.6
# requests==2.31.0

import logging
import os
import secrets
import time
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Query, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from jose import JWTError, jwt
from passlib.context import CryptContext
import psycopg2
from psycopg2.extras import RealDictCursor
import requests

from fuzzy_allocator import (
    fetch_installer_historical_feature_stats,
    normalize_installer_name,
    score_installer_with_fuzzy_logic,
)
from ml_model import InstallerMLModel

logger = logging.getLogger("lead_allocation")

# Load environment variables from a project-level .env file when running locally
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# ============================================
# CONFIGURATION
# ============================================

class Settings:
    #DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:4567@localhost:5432/capstone25")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://q4gems_admin:890*()iopIOP@capstone25.postgres.database.azure.com:5432/capstone25db")  #Azure specific
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours
    ML_PUBLIC_API_KEY = os.getenv("ML_PUBLIC_API_KEY")
    
    # Geocoding API (Google Maps or alternative)
    GEOCODING_API_KEY = os.getenv("GEOCODING_API_KEY", "")
    GEOCODING_PROVIDER = os.getenv("GEOCODING_PROVIDER", "nominatim")  # 'google' or 'nominatim'

settings = Settings()

# ============================================
# FASTAPI APP SETUP
# ============================================

app = FastAPI(
    title="Lead Allocation System API",
    description="Intelligent lead allocation for window film installers",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/admin/login")

# ============================================
# DATABASE CONNECTION
# ============================================

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def execute_query(query: str, params: tuple = None, fetch: bool = True):
    """Execute database query with error handling"""
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


def resolve_final_installer_selection(record: Dict[str, Any]) -> Optional[str]:
    """Return the most accurate final installer name for a record."""

    preferred = (record.get("final_installer_selection") or "").strip()
    if preferred:
        return preferred

    for field in ("assigned_installer_name", "installer_name", "dealer_name"):
        candidate = (record.get(field) or "").strip()
        if candidate:
            return candidate

    return None


def insert_historical_record_from_lead(lead: Dict[str, Any]) -> None:
    """Persist a converted lead into historical_data for future ML training."""

    final_installer = resolve_final_installer_selection(lead)
    dealer_name = lead.get("dealer_name") or final_installer or lead.get("assigned_installer_name")

    query = """
        INSERT INTO historical_data (
            submit_date,
            first_name,
            address1,
            city,
            province,
            postal,
            dealer_name,
            project_type,
            current_status,
            final_installer_selection,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """

    params = (
        lead.get("created_at"),
        lead.get("name"),
        lead.get("address"),
        lead.get("city"),
        lead.get("province"),
        lead.get("postal_code"),
        dealer_name,
        lead.get("job_type"),
        lead.get("status"),
        final_installer,
    )

    execute_query(query, params, fetch=False)


# Initialize the ML allocator once so it can be reused across requests
ml_allocator = InstallerMLModel(execute_query)

# Distance preferences used when evaluating installer suitability
PREFERRED_DISTANCE_KM = 120
ALTERNATIVE_DISTANCE_LIMIT_KM = 50
LOCAL_PRIORITY_DISTANCE_KM = 75
FUZZY_MAX_DISTANCE_KM = 200
ABSOLUTE_DISTANCE_LIMIT_KM = 400

# Lightweight in-memory cache for installer pools per province
INSTALLER_CACHE_TTL = timedelta(minutes=5)
_INSTALLER_CACHE: Dict[str, Dict[str, Any]] = {}

# ============================================
# PYDANTIC MODELS
# ============================================

class LeadCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=20)
    address: str = Field(..., min_length=5, max_length=255)
    city: str = Field(..., min_length=2, max_length=100)
    province: str = Field(..., min_length=2, max_length=50)
    postal_code: Optional[str] = Field(None, max_length=10)
    job_type: str = Field(..., pattern="^(residential|commercial)$")
    comments: Optional[str] = None

    @validator('province')
    def validate_province(cls, v):
        valid_provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
        if v.upper() not in valid_provinces:
            raise ValueError(f'Province must be one of: {", ".join(valid_provinces)}')
        return v.upper()

class AlternativeInstaller(BaseModel):
    id: int
    name: str
    city: str
    province: str
    distance_km: float
    allocation_score: float
    active_leads: int
    converted_leads: Optional[int] = None
    ml_probability: Optional[float] = None
    distance_review_required: Optional[bool] = None

class InstallerOverrideRequest(BaseModel):
    installer_id: Optional[int] = None

class LeadResponse(BaseModel):
    id: int
    name: str
    email: str
    city: str
    province: str
    job_type: str
    status: str
    assigned_installer_id: Optional[int]
    assigned_installer_name: Optional[str]
    final_installer_selection: Optional[str]
    allocation_score: Optional[float]
    distance_to_installer_km: Optional[float]
    installer_ml_probability: Optional[float] = None
    distance_review_required: Optional[bool] = None
    alternative_installers: Optional[List[AlternativeInstaller]]
    created_at: datetime
    message: str = "Lead submitted successfully"


class MLStatusResponse(BaseModel):
    trained: bool
    last_trained_at: Optional[datetime]
    training_rows: Optional[int]
    last_error: Optional[str]
    message: str = "ok"


class MLTrainRequest(BaseModel):
    force: bool = True

class AdminUser(BaseModel):
    id: int
    username: str
    email: str
    last_name: Optional[str]
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: AdminUser

class DashboardStats(BaseModel):
    total_leads: int
    pending_leads: int
    assigned_leads: int
    completed_leads: int
    conversion_rate: float
    avg_allocation_score: float
    active_installers: int

# ============================================
# HAVERSINE DISTANCE CALCULATION
# ============================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Radius of earth in kilometers
    R = 6371
    
    return R * c

# ============================================
# GEOCODING FUNCTIONS
# ============================================

def geocode_address(address: str, city: str, province: str) -> tuple:
    """
    Convert address to latitude/longitude coordinates
    Returns (latitude, longitude) or raises exception
    """
    full_address = f"{address}, {city}, {province}, Canada"
    
    if settings.GEOCODING_PROVIDER == "google" and settings.GEOCODING_API_KEY:
        # Google Maps Geocoding API
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": full_address,
            "key": settings.GEOCODING_API_KEY
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data["status"] == "OK" and data["results"]:
                location = data["results"][0]["geometry"]["location"]
                return (location["lat"], location["lng"])
            else:
                raise Exception(f"Geocoding failed: {data.get('status', 'Unknown error')}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Geocoding error: {str(e)}")
    
    else:
        # Free Nominatim API (OpenStreetMap)
        # Nominatim requires 1 second between requests
        time.sleep(1)
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": full_address,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "LeadAllocationSystem/1.0"
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            data = response.json()
            
            if data and len(data) > 0:
                return (float(data[0]["lat"]), float(data[0]["lon"]))
            else:
                raise Exception("Address not found")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Geocoding error: {str(e)}")


def build_ml_feature_payload(source: Optional[Any]) -> dict:
    """Extract ML-specific attributes from a lead or record."""

    if source is None:
        return {}

    if isinstance(source, dict):
        payload = source
    elif hasattr(source, "model_dump"):
        payload = source.model_dump()
    else:
        payload = getattr(source, "__dict__", {})

    return {
        "project_type": payload.get("project_type") or payload.get("job_type"),
        "square_footage": payload.get("square_footage") or payload.get("square_feet"),
        "current_status": payload.get("current_status") or payload.get("status"),
    }

# ============================================
# LEAD ALLOCATION ALGORITHM - ENHANCED VERSION
# ============================================

def fetch_active_installers_by_province(province: str) -> List[Dict[str, Any]]:
    """Return active installers for the provided province (with caching)."""

    now = datetime.utcnow()
    cached_entry = _INSTALLER_CACHE.get(province)
    if cached_entry and cached_entry["expires_at"] > now:
        return deepcopy(cached_entry["payload"])

    query = """
        SELECT
            i.id,
            i.name,
            i.email,
            i.city,
            i.province,
            i.latitude,
            i.longitude,
            i.is_active,
            COUNT(CASE WHEN l.status = 'active' THEN 1 END) as active_leads,
            COUNT(CASE WHEN l.status = 'converted' THEN 1 END) as converted_leads
        FROM installers i
        LEFT JOIN leads l ON i.id = l.assigned_installer_id
        WHERE i.is_active = TRUE
            AND i.province = %s
            AND i.latitude IS NOT NULL
            AND i.longitude IS NOT NULL
        GROUP BY i.id
    """

    installers = execute_query(query, (province,))
    _INSTALLER_CACHE[province] = {
        "payload": deepcopy(installers),
        "expires_at": now + INSTALLER_CACHE_TTL,
    }
    return installers


def allocate_lead_to_installer(
    lead_lat: float,
    lead_lon: float,
    province: str,
    lead_payload: Optional[Any] = None,
    installer_pool: Optional[List[Dict[str, Any]]] = None,
    historical_stats: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Enhanced allocation algorithm that returns:
    - Best installer (highest composite score)
    - 2-3 alternative installers within 50km

    Composite score blends:
    - ML probability learned from historical data
    - Geographic distance to the dealer
    - Closed deals and currently active allocations
    """
    
    # Get all active installers from the same province
    if installer_pool is not None:
        installers = installer_pool
    else:
        installers = fetch_active_installers_by_province(province)

    if not installers:
        return None

    if historical_stats is None:
        historical_stats = fetch_installer_historical_feature_stats(execute_query)

    # Prepare ML probabilities for this lead (still exposed for debugging)
    lead_features = build_ml_feature_payload(lead_payload)
    ml_probabilities = ml_allocator.predict_probabilities(lead_features)

    scored_installers = []
    max_closed = max((installer.get('converted_leads') or 0) for installer in installers) if installers else 0
    max_active = max((installer.get('active_leads') or 0) for installer in installers) if installers else 0

    for installer in installers:
        distance_km = haversine_distance(
            lead_lat, lead_lon,
            installer['latitude'], installer['longitude']
        )

        if distance_km > ABSOLUTE_DISTANCE_LIMIT_KM:
            continue

        normalized_name = normalize_installer_name(installer.get('name'))
        installer_stats = historical_stats.get(normalized_name, {}) if normalized_name else {}

        fuzzy_score, breakdown = score_installer_with_fuzzy_logic(
            distance_km=distance_km,
            installer_stats=installer_stats,
            lead_features=lead_features,
        )

        probability = (
            ml_probabilities.get(installer['name'])
            or ml_probabilities.get((installer.get('name') or '').lower())
            or 0.0
        )
        closed_leads = installer.get('converted_leads') or 0
        active_leads = installer.get('active_leads') or 0

        conversion_component = (closed_leads / max_closed) if max_closed else 0
        workload_component = (active_leads / max_active) if max_active else 0

        allocation_score = fuzzy_score
        allocation_score += conversion_component * 0.05
        allocation_score -= workload_component * 0.05
        allocation_score = max(0.0, min(1.0, allocation_score))

        if distance_km <= LOCAL_PRIORITY_DISTANCE_KM:
            distance_bucket = 0
        elif distance_km <= PREFERRED_DISTANCE_KM:
            distance_bucket = 1
        elif distance_km <= FUZZY_MAX_DISTANCE_KM:
            distance_bucket = 2
        else:
            distance_bucket = 3

        scored_installers.append({
            'installer_id': installer['id'],
            'installer_name': installer['name'],
            'city': installer['city'],
            'province': installer['province'],
            'distance_km': round(distance_km, 2),
            'allocation_score': round(allocation_score, 4),
            'active_leads': active_leads,
            'converted_leads': closed_leads,
            'ml_probability': round(probability, 4),
            'distance_review_required': distance_km > FUZZY_MAX_DISTANCE_KM,
            'distance_bucket': distance_bucket,
            'score_breakdown': breakdown,
        })

    if not scored_installers:
        return None

    scored_installers.sort(key=lambda x: (x['distance_bucket'], -x['allocation_score']))

    prioritized = [
        installer for installer in scored_installers
        if installer['distance_km'] <= FUZZY_MAX_DISTANCE_KM
    ] or scored_installers

    def sanitize(installer: dict) -> dict:
        cleaned = dict(installer)
        cleaned.pop('distance_bucket', None)
        cleaned.pop('score_breakdown', None)

        installer_id = cleaned.get('installer_id')
        installer_name = cleaned.get('installer_name')

        # Provide both legacy keys (installer_id/installer_name) and the new
        # consumer-friendly aliases (id/name) so the frontend dropdown can rely
        # on a stable schema without breaking existing references.
        if installer_id is not None:
            cleaned.setdefault('id', installer_id)
        if installer_name:
            cleaned.setdefault('name', installer_name)

        return cleaned

    best_installer = sanitize(prioritized[0])

    alternatives = [
        sanitize(installer)
        for installer in scored_installers
        if installer['installer_id'] != best_installer['installer_id']
        and installer['distance_km'] <= ALTERNATIVE_DISTANCE_LIMIT_KM
    ][:3]

    return {
        'best_installer': best_installer,
        'alternative_installers': alternatives
    }

# ============================================
# AUTHENTICATION & AUTHORIZATION
# ============================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def resolve_admin_user_from_token(token: str) -> Optional[AdminUser]:
    """Return an AdminUser from a JWT token, or None if invalid."""

    if not token:
        return None

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if not username:
            return None
    except JWTError:
        return None

    query = "SELECT id, username, email, last_name, role FROM admin_users WHERE username = %s AND is_active = TRUE"
    user = execute_query(query, (username,))

    if not user:
        return None

    return AdminUser(**user[0])


async def get_current_user(token: str = Depends(oauth2_scheme)) -> AdminUser:
    """Require a valid authenticated user."""

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = resolve_admin_user_from_token(token)
    if not user:
        raise credentials_exception

    return user


def has_valid_ml_api_key(provided_key: Optional[str]) -> bool:
    """Verify the optional API key for ML endpoints."""

    return bool(
        provided_key
        and settings.ML_PUBLIC_API_KEY
        and secrets.compare_digest(provided_key, settings.ML_PUBLIC_API_KEY)
    )

# ============================================
# API ENDPOINTS - PUBLIC
# ============================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Lead Allocation System API is running"}

@app.post("/api/leads", response_model=LeadResponse)
async def create_lead(lead: LeadCreate):
    """
    Public endpoint - Submit a new lead
    Automatically assigns to best installer using enhanced ML algorithm
    Returns alternative installer options
    """
    
    try:
        # Step 1: Geocode the lead address
        try:
            lead_lat, lead_lon = geocode_address(lead.address, lead.city, lead.province)
        except HTTPException as he:
            # If geocoding fails, still create the lead but don't assign installer
            lead_lat, lead_lon = None, None
        
        # Step 2: Get allocation (best + alternatives) if geocoding succeeded
        allocation = None
        if lead_lat and lead_lon:
            allocation = allocate_lead_to_installer(lead_lat, lead_lon, lead.province, lead)
        
        # Step 3: Insert lead into database
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                assigned_installer_id = allocation['best_installer']['installer_id'] if allocation else None
                final_installer_selection = allocation['best_installer']['installer_name'] if allocation else None
                query = """
                    INSERT INTO leads (
                        name, email, phone, address, city, province, postal_code,
                        job_type, comments, status, assigned_installer_id,
                        allocation_score, distance_to_installer_km,
                        final_installer_selection,
                        latitude, longitude, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING id, created_at
                """

                cursor.execute(query, (
                    lead.name,
                    lead.email,
                    lead.phone,
                    lead.address,
                    lead.city,
                    lead.province,
                    lead.postal_code,
                    lead.job_type,
                    lead.comments,
                    'active',
                    assigned_installer_id,
                    allocation['best_installer']['allocation_score'] if allocation else None,
                    allocation['best_installer']['distance_km'] if allocation else None,
                    final_installer_selection,
                    lead_lat,
                    lead_lon
                ))
                
                result = cursor.fetchone()
                lead_id = result['id']
                created_at = result['created_at']
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to insert lead: {str(e)}")
        finally:
            conn.close()
        
        # Step 4: Format alternative installers response
        alternative_installers = None
        if allocation and allocation['alternative_installers']:
            alternative_installers = [
                AlternativeInstaller(
                    id=alt['installer_id'],
                    name=alt['installer_name'],
                    city=alt['city'],
                    province=alt['province'],
                    distance_km=alt['distance_km'],
                    allocation_score=alt['allocation_score'],
                    active_leads=alt['active_leads'],
                    converted_leads=alt.get('converted_leads'),
                    ml_probability=alt.get('ml_probability'),
                    distance_review_required=alt.get('distance_review_required'),
                ) for alt in allocation['alternative_installers']
            ]
        
        # Step 5: Return response
        return LeadResponse(
            id=lead_id,
            name=lead.name,
            email=lead.email,
            city=lead.city,
            province=lead.province,
            job_type=lead.job_type,
            status='active',
            assigned_installer_id=allocation['best_installer']['installer_id'] if allocation else None,
            assigned_installer_name=allocation['best_installer']['installer_name'] if allocation else None,
            final_installer_selection=allocation['best_installer']['installer_name'] if allocation else None,
            allocation_score=allocation['best_installer']['allocation_score'] if allocation else None,
            distance_to_installer_km=allocation['best_installer']['distance_km'] if allocation else None,
            installer_ml_probability=allocation['best_installer'].get('ml_probability') if allocation else None,
            distance_review_required=allocation['best_installer'].get('distance_review_required') if allocation else None,
            alternative_installers=alternative_installers,
            created_at=created_at,
            message="Lead submitted and assigned successfully" if allocation else "Lead submitted - no installer available in your area"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing lead: {str(e)}")

# ============================================
# API ENDPOINTS - ADMIN (PROTECTED)
# ============================================

@app.post("/api/admin/login", response_model=Token)
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Admin login endpoint"""
    
    query = "SELECT id, username, email, last_name, role, password_hash FROM admin_users WHERE username = %s AND is_active = TRUE"
    user = execute_query(query, (form_data.username,))
    
    if not user or not verify_password(form_data.password, user[0]['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    execute_query(
        "UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
        (user[0]['id'],),
        fetch=False
    )
    
    # Create access token
    access_token = create_access_token(data={"sub": user[0]['username']})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=AdminUser(
            id=user[0]['id'],
            username=user[0]['username'],
            email=user[0]['email'],
            last_name=user[0]['last_name'],
            role=user[0]['role']
        )
    )

@app.get("/api/admin/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(current_user: AdminUser = Depends(get_current_user)):
    """Get dashboard statistics"""
    
    query = """
        SELECT 
            COUNT(*) as total_leads,
            COUNT(CASE WHEN status = 'active' THEN 1 END) as pending_leads,
            COUNT(CASE WHEN assigned_installer_id IS NOT NULL THEN 1 END) as assigned_leads,
            COUNT(CASE WHEN status = 'converted' THEN 1 END) as completed_leads,
            AVG(allocation_score) as avg_allocation_score,
            (COUNT(CASE WHEN status = 'converted' THEN 1 END)::float / 
             NULLIF(COUNT(CASE WHEN status IN ('converted', 'dead') THEN 1 END), 0) * 100) as conversion_rate
        FROM leads
    """
    
    stats = execute_query(query)[0]
    
    # Get active installers count
    active_installers = execute_query("SELECT COUNT(*) as count FROM installers WHERE is_active = TRUE")[0]['count']
    
    return DashboardStats(
        total_leads=stats['total_leads'] or 0,
        pending_leads=stats['pending_leads'] or 0,
        assigned_leads=stats['assigned_leads'] or 0,
        completed_leads=stats['completed_leads'] or 0,
        conversion_rate=round(stats['conversion_rate'] or 0, 2),
        avg_allocation_score=round(stats['avg_allocation_score'] or 0, 2),
        active_installers=active_installers
    )

@app.get("/api/admin/leads")
async def get_all_leads(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: AdminUser = Depends(get_current_user)
):
    """Get all leads with optional status filter - includes alternative installers"""
    
    if status:
        query = """
            SELECT l.*, i.name as installer_name, i.city as installer_city
            FROM leads l
            LEFT JOIN installers i ON l.assigned_installer_id = i.id
            WHERE l.status = %s
            ORDER BY l.created_at DESC
            LIMIT %s OFFSET %s
        """
        params = (status, limit, offset)
    else:
        query = """
            SELECT l.*, i.name as installer_name, i.city as installer_city
            FROM leads l
            LEFT JOIN installers i ON l.assigned_installer_id = i.id
            ORDER BY l.created_at DESC
            LIMIT %s OFFSET %s
        """
        params = (limit, offset)
    
    leads = execute_query(query, params)
    
    # Cache installers per province within this request to avoid repetitive queries
    installer_cache: Dict[str, List[Dict[str, Any]]] = {}
    try:
        historical_stats = fetch_installer_historical_feature_stats(execute_query)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Unable to preload historical installer stats: %s", exc)
        historical_stats = None

    # For each lead, calculate alternative installers if coordinates exist
    enhanced_leads = []
    for lead in leads:
        lead_dict = dict(lead)

        allocation = None
        lead_lat = lead.get('latitude')
        lead_lon = lead.get('longitude')
        should_score_alternatives = (
            lead.get('status') == 'active'
            and lead_lat is not None
            and lead_lon is not None
        )
        if should_score_alternatives:
            province_key = lead.get('province')
            installer_pool: Optional[List[Dict[str, Any]]] = None
            if province_key:
                if province_key not in installer_cache:
                    try:
                        installer_cache[province_key] = fetch_active_installers_by_province(province_key)
                    except HTTPException as exc:
                        logger.warning(
                            "Installer lookup failed for province %s: %s",
                            province_key,
                            getattr(exc, 'detail', str(exc)),
                        )
                        installer_cache[province_key] = []
                installer_pool = installer_cache.get(province_key)
            try:
                allocation = allocate_lead_to_installer(
                    lead_lat,
                    lead_lon,
                    lead.get('province'),
                    lead,
                    installer_pool=installer_pool,
                    historical_stats=historical_stats,
                )
            except HTTPException as exc:
                logger.warning(
                    "Allocation preview failed for lead %s: %s",
                    lead.get('id'),
                    getattr(exc, 'detail', str(exc)),
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Unexpected allocation failure for lead %s", lead.get('id'))

        if allocation:
            lead_dict['installer_ml_probability'] = allocation['best_installer'].get('ml_probability')
            lead_dict['distance_review_required'] = allocation['best_installer'].get('distance_review_required')
            lead_dict['alternative_installers'] = allocation['alternative_installers']
        else:
            lead_dict['alternative_installers'] = []
            lead_dict['distance_review_required'] = None

        resolved_final = resolve_final_installer_selection(lead_dict)
        if resolved_final:
            lead_dict['final_installer_selection'] = resolved_final

        enhanced_leads.append(lead_dict)
    
    # Also get total count
    count_query = "SELECT COUNT(*) as total FROM leads"
    if status:
        count_query += " WHERE status = %s"
        total_count = execute_query(count_query, (status,))[0]['total']
    else:
        total_count = execute_query(count_query)[0]['total']
    
    return {"leads": enhanced_leads, "count": len(enhanced_leads), "total": total_count}

@app.get("/api/admin/leads/{lead_id}")
async def get_lead_detail(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Get detailed information about a specific lead - includes alternative installers"""
    
    query = """
        SELECT l.*, 
               i.name as installer_name, 
               i.email as installer_email,
               i.phone as installer_phone,
               i.city as installer_city
        FROM leads l
        LEFT JOIN installers i ON l.assigned_installer_id = i.id
        WHERE l.id = %s
    """
    
    lead = execute_query(query, (lead_id,))
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    lead_dict = dict(lead[0])
    
    allocation = None
    if lead_dict['latitude'] and lead_dict['longitude']:
        try:
            allocation = allocate_lead_to_installer(
                lead_dict['latitude'],
                lead_dict['longitude'],
                lead_dict['province'],
                lead_dict
            )
        except HTTPException as exc:
            logger.warning(
                "Allocation preview failed for lead detail %s: %s",
                lead_id,
                getattr(exc, 'detail', str(exc)),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected allocation failure for lead %s", lead_id)

    if allocation:
        lead_dict['installer_ml_probability'] = allocation['best_installer'].get('ml_probability')
        lead_dict['distance_review_required'] = allocation['best_installer'].get('distance_review_required')
        lead_dict['alternative_installers'] = allocation['alternative_installers']
    else:
        lead_dict['alternative_installers'] = []
        lead_dict['distance_review_required'] = None

    resolved_final = resolve_final_installer_selection(lead_dict)
    if resolved_final:
        lead_dict['final_installer_selection'] = resolved_final

    return lead_dict

@app.patch("/api/admin/leads/{lead_id}/status")
async def update_lead_status(
    lead_id: int,
    status: str,
    current_user: AdminUser = Depends(get_current_user)
):
    """Update lead status"""
    
    valid_statuses = ['active', 'converted', 'dead', 'follow_up']
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
    
    lead_rows = execute_query(
        """
        SELECT l.*, i.name as assigned_installer_name
        FROM leads l
        LEFT JOIN installers i ON l.assigned_installer_id = i.id
        WHERE l.id = %s
        """,
        (lead_id,),
    )

    if not lead_rows:
        raise HTTPException(status_code=404, detail="Lead not found")

    current_lead = lead_rows[0]
    previous_status = current_lead.get('status')

    query = "UPDATE leads SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
    execute_query(query, (status, lead_id), fetch=False)

    if status == 'converted' and previous_status != 'converted':
        lead_copy: Dict[str, Any] = dict(current_lead)
        lead_copy['status'] = status
        insert_historical_record_from_lead(lead_copy)

    return {"message": "Lead status updated successfully", "lead_id": lead_id, "new_status": status}

@app.patch("/api/admin/leads/{lead_id}/installer-override")
async def update_installer_override(
    lead_id: int,
    override: Optional[InstallerOverrideRequest] = None,
    installer_id: Optional[int] = Query(default=None),
    current_user: AdminUser = Depends(get_current_user)
):
    """Update installer override - allows manual assignment of alternative installers"""

    resolved_installer_id = installer_id
    if override and override.installer_id is not None:
        resolved_installer_id = override.installer_id

    lead_rows = execute_query(
        """
        SELECT l.id,
               l.assigned_installer_id,
               l.installer_override_id,
               l.final_installer_selection,
               i.name AS assigned_installer_name,
               i.city AS assigned_installer_city
        FROM leads l
        LEFT JOIN installers i ON l.assigned_installer_id = i.id
        WHERE l.id = %s
        """,
        (lead_id,),
    )
    if not lead_rows:
        raise HTTPException(status_code=404, detail="Lead not found")

    lead_record = lead_rows[0]
    assigned_installer_id = lead_record.get('assigned_installer_id')
    assigned_installer_name = lead_record.get('assigned_installer_name')
    assigned_installer_city = lead_record.get('assigned_installer_city')
    final_installer_name = resolve_final_installer_selection(lead_record)

    if assigned_installer_id and (not assigned_installer_name or not assigned_installer_city):
        installer_name_row = execute_query(
            "SELECT name, city FROM installers WHERE id = %s",
            (assigned_installer_id,),
        )
        if installer_name_row:
            assigned_installer_name = installer_name_row[0]['name']
            assigned_installer_city = installer_name_row[0].get('city')

    # Validate installer exists if provided and capture its name
    if resolved_installer_id:
        installer_check = execute_query(
            "SELECT id, name, city FROM installers WHERE id = %s AND is_active = TRUE",
            (resolved_installer_id,)
        )
        if not installer_check:
            raise HTTPException(status_code=404, detail="Installer not found or inactive")
        assigned_installer_id = installer_check[0]['id']
        assigned_installer_name = installer_check[0]['name']
        assigned_installer_city = installer_check[0].get('city')
        final_installer_name = installer_check[0]['name']
    elif assigned_installer_id:
        final_installer_name = assigned_installer_name
    else:
        final_installer_name = None

    query = """
        UPDATE leads
        SET installer_override_id = %s,
            assigned_installer_id = %s,
            final_installer_selection = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
    """
    execute_query(query, (resolved_installer_id, assigned_installer_id, final_installer_name, lead_id), fetch=False)

    logger.info(
        "Lead %s installer override updated by %s (override_id=%s, final_installer=%s)",
        lead_id,
        current_user.username,
        resolved_installer_id,
        final_installer_name,
    )

    return {
        "message": "Installer override updated successfully",
        "lead_id": lead_id,
        "installer_id": resolved_installer_id,
        "assigned_installer_id": assigned_installer_id,
        "final_installer_selection": final_installer_name,
        "installer_name": assigned_installer_name,
        "installer_city": assigned_installer_city,
    }

@app.get("/api/admin/installers")
async def get_installers(current_user: AdminUser = Depends(get_current_user)):
    """Get all installers with performance metrics"""
    
    query = """
        SELECT * FROM installer_performance
        ORDER BY province, city
    """
    
    installers = execute_query(query)
    return {"installers": installers, "count": len(installers)}

@app.get("/api/admin/historical-data")
async def get_historical_data(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    current_user: AdminUser = Depends(get_current_user)
):
    """Get historical data records"""
    
    if status and status != 'all':
        query = """
            SELECT * FROM historical_data
            WHERE current_status = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        params = (status, limit, offset)
    else:
        query = """
            SELECT * FROM historical_data
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        params = (limit, offset)
    
    data = execute_query(query, params)
    
    # Get total count
    count_query = "SELECT COUNT(*) as total FROM historical_data"
    if status and status != 'all':
        count_query += " WHERE current_status = %s"
        total_count = execute_query(count_query, (status,))[0]['total']
    else:
        total_count = execute_query(count_query)[0]['total']
    
    return {"data": data, "count": len(data), "total": total_count}


def extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    """Parse an Authorization header and return the bearer token when present."""

    if not auth_header:
        return None

    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]

    return None


def resolve_request_user(
    token_param: Optional[str],
    auth_header: Optional[str],
) -> Optional[AdminUser]:
    """Resolve an AdminUser from either the query token or Authorization header."""

    token_value = token_param or extract_bearer_token(auth_header)
    if not token_value:
        return None

    return resolve_admin_user_from_token(token_value)


def resolve_request_api_key(
    query_api_key: Optional[str],
    header_api_key: Optional[str],
) -> Optional[str]:
    """Return whichever API key source is populated (query param or header)."""

    return query_api_key or header_api_key


@app.get("/api/admin/ml/status", response_model=MLStatusResponse)
async def get_ml_status(
    api_key: Optional[str] = Query(None, alias="api_key"),
    token: Optional[str] = Query(None, alias="token"),
    header_api_key: Optional[str] = Header(None, alias="X-ML-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Return the current ML training status for administrators or API key holders."""

    resolved_user = resolve_request_user(token, authorization)
    provided_key = resolve_request_api_key(api_key, header_api_key)
    if not (resolved_user or has_valid_ml_api_key(provided_key)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    status_payload = ml_allocator.status()
    return MLStatusResponse(
        **status_payload,
        message="ok" if status_payload.get("trained") else "not_trained",
    )


@app.api_route("/api/admin/ml/train", response_model=MLStatusResponse, methods=["GET", "POST"])
async def trigger_ml_training(
    payload: Optional[MLTrainRequest] = None,
    force: Optional[bool] = Query(None, alias="force"),
    api_key: Optional[str] = Query(None, alias="api_key"),
    token: Optional[str] = Query(None, alias="token"),
    header_api_key: Optional[str] = Header(None, alias="X-ML-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Allow administrators (or requests with the ML API key) to retrain the model on demand."""

    resolved_user = resolve_request_user(token, authorization)
    provided_key = resolve_request_api_key(api_key, header_api_key)
    if not (resolved_user or has_valid_ml_api_key(provided_key)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    force_value: Optional[bool] = None
    if payload is not None:
        force_value = payload.force
    if force is not None:
        force_value = force
    if force_value is None:
        force_value = True

    success = ml_allocator.train(force=force_value)
    status_payload = ml_allocator.status()
    message = "trained" if success else "training_failed"
    return MLStatusResponse(**status_payload, message=message)

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)