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
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from jose import JWTError, jwt
from passlib.context import CryptContext
import psycopg2
from psycopg2.extras import RealDictCursor
import requests

from dealer_allocator import (
    fetch_dealer_historical_feature_stats,
    normalize_dealer_name,
    score_dealer_with_fuzzy_logic,
)
from dealer_ml_model import DealerMLModel
from mcp_server import mcp_app

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
    description="Intelligent lead allocation for window film dealers",
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
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================
# REGISTER API ROUTE MODULES
# ============================================
from routes.admin_leads import router as admin_leads_router
from routes.admin_dealers import router as admin_dealers_router
from routes.admin_history import router as admin_history_router
from routes.admin_logs import router as admin_logs_router
from routes.admin_audit_log import router as admin_audit_log_router
from routes.admin_cost_tracking import router as admin_cost_tracking_router
from routes.admin_ai_controls import router as admin_ai_controls_router
from routes.admin_lasso_dashboard import router as admin_lasso_dashboard_router
from routes.admin_reports import router as admin_reports_router
from routes.admin_resources import router as admin_resources_router
from routes.admin_tools import router as admin_tools_router
from routes.ai_leads import router as ai_leads_router
from routes.ai_insights import router as ai_insights_router
from routes.dealer import router as dealer_router
from routes.email_intel import router as email_intel_router

app.include_router(admin_leads_router)
app.include_router(admin_dealers_router)
app.include_router(admin_history_router)
app.include_router(admin_logs_router)
app.include_router(admin_audit_log_router)
app.include_router(admin_cost_tracking_router)
app.include_router(admin_ai_controls_router)
app.include_router(admin_lasso_dashboard_router)
app.include_router(admin_reports_router)
app.include_router(admin_resources_router)
app.include_router(admin_tools_router)
app.include_router(ai_leads_router)
app.include_router(ai_insights_router)
app.include_router(dealer_router)
app.include_router(email_intel_router)
app.mount("/mcp", mcp_app)

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


def resolve_final_dealer_selection(record: Dict[str, Any]) -> Optional[str]:
    """Return the most accurate final dealer name for a record."""

    preferred = (record.get("final_dealer_selection") or "").strip()
    if preferred:
        return preferred

    for field in ("dealer_name_assigned", "recommended_dealer_name", "dealer_name"):
        candidate = (record.get(field) or "").strip()
        if candidate:
            return candidate

    return None


def sync_lead_to_historical(lead_id: int, executor=None) -> None:
    """Historical persistence is retired in Lasso-only mode."""
    return None


# Initialize the ML allocator once so it can be reused across requests
ml_allocator = DealerMLModel(execute_query)

# Distance preferences used when evaluating dealer suitability
PREFERRED_DISTANCE_KM = 120
ALTERNATIVE_DISTANCE_LIMIT_KM = 50
LOCAL_PRIORITY_DISTANCE_KM = 75
FUZZY_MAX_DISTANCE_KM = 200
ABSOLUTE_DISTANCE_LIMIT_KM = 400

# Lightweight in-memory cache for dealer pools per province
DEALER_CACHE_TTL = timedelta(minutes=5)
_DEALER_CACHE: Dict[str, Dict[str, Any]] = {}

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

class AlternativeDealer(BaseModel):
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

class LeadResponse(BaseModel):
    id: int
    name: str
    email: str
    city: str
    province: str
    job_type: str
    status: str
    assigned_dealer_id: Optional[int]
    dealer_name_assigned: Optional[str]
    recommended_dealer_id: Optional[int]
    recommended_dealer_name: Optional[str]
    final_dealer_selection: Optional[str]
    allocation_score: Optional[float]
    distance_to_dealer_km: Optional[float]
    dealer_ml_probability: Optional[float] = None
    distance_review_required: Optional[bool] = None
    alternative_dealers: Optional[List[AlternativeDealer]]
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
    active_dealers: int

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

def fetch_active_dealers_by_province(province: str) -> List[Dict[str, Any]]:
    """Return active dealers for the provided province, with caching."""

    now = datetime.utcnow()
    cached_entry = _DEALER_CACHE.get(province)
    if cached_entry and cached_entry["expires_at"] > now:
        return deepcopy(cached_entry["payload"])

    query = """
        SELECT
            d.id,
            d.name,
            d.email,
            d.city,
            d.province,
            d.latitude,
            d.longitude,
            d.is_active,
            COUNT(CASE WHEN l.status = 'active' THEN 1 END) AS active_leads,
            COUNT(CASE WHEN l.status = 'converted' THEN 1 END) AS converted_leads
        FROM dealers d
        LEFT JOIN leads l ON d.id = l.assigned_dealer_id
        WHERE d.is_active = TRUE
          AND d.province = %s
          AND d.latitude IS NOT NULL
          AND d.longitude IS NOT NULL
        GROUP BY d.id
    """

    dealers = execute_query(query, (province,))
    _DEALER_CACHE[province] = {
        "payload": deepcopy(dealers),
        "expires_at": now + DEALER_CACHE_TTL,
    }
    return dealers


def allocate_lead_to_dealer(
    lead_lat: float,
    lead_lon: float,
    province: str,
    lead_payload: Optional[Any] = None,
    dealer_pool: Optional[List[Dict[str, Any]]] = None,
    historical_stats: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Enhanced allocation algorithm that returns the best dealer plus alternatives.

    Composite score blends:
    - ML probability learned from historical data
    - Geographic distance
    - Closed deals and currently active allocations
    """

    dealers = dealer_pool if dealer_pool is not None else fetch_active_dealers_by_province(province)
    if not dealers:
        return None

    if historical_stats is None:
        historical_stats = fetch_dealer_historical_feature_stats(execute_query)

    lead_features = build_ml_feature_payload(lead_payload)
    ml_probabilities = ml_allocator.predict_probabilities(lead_features)

    scored_dealers = []
    max_closed = max((dealer.get("converted_leads") or 0) for dealer in dealers) if dealers else 0
    max_active = max((dealer.get("active_leads") or 0) for dealer in dealers) if dealers else 0

    for dealer in dealers:
        distance_km = haversine_distance(
            lead_lat,
            lead_lon,
            dealer["latitude"],
            dealer["longitude"],
        )
        if distance_km > ABSOLUTE_DISTANCE_LIMIT_KM:
            continue

        normalized_name = normalize_dealer_name(dealer.get("name"))
        dealer_stats = historical_stats.get(normalized_name, {}) if normalized_name else {}
        fuzzy_score, breakdown = score_dealer_with_fuzzy_logic(
            distance_km=distance_km,
            dealer_stats=dealer_stats,
            lead_features=lead_features,
        )

        probability = (
            ml_probabilities.get(dealer["name"])
            or ml_probabilities.get((dealer.get("name") or "").lower())
            or 0.0
        )
        closed_leads = dealer.get("converted_leads") or 0
        active_leads = dealer.get("active_leads") or 0

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

        scored_dealers.append(
            {
                "dealer_id": dealer["id"],
                "dealer_name": dealer["name"],
                "city": dealer["city"],
                "province": dealer["province"],
                "distance_km": round(distance_km, 2),
                "allocation_score": round(allocation_score, 4),
                "active_leads": active_leads,
                "converted_leads": closed_leads,
                "ml_probability": round(probability, 4),
                "distance_review_required": distance_km > FUZZY_MAX_DISTANCE_KM,
                "distance_bucket": distance_bucket,
                "score_breakdown": breakdown,
            }
        )

    if not scored_dealers:
        return None

    scored_dealers.sort(key=lambda row: (row["distance_bucket"], -row["allocation_score"]))
    prioritized = [
        dealer for dealer in scored_dealers if dealer["distance_km"] <= FUZZY_MAX_DISTANCE_KM
    ] or scored_dealers

    def sanitize(dealer: dict) -> dict:
        cleaned = dict(dealer)
        cleaned.pop("distance_bucket", None)
        cleaned.pop("score_breakdown", None)
        if cleaned.get("dealer_id") is not None:
            cleaned.setdefault("id", cleaned["dealer_id"])
        if cleaned.get("dealer_name"):
            cleaned.setdefault("name", cleaned["dealer_name"])
        return cleaned

    best_dealer = sanitize(prioritized[0])
    alternatives = [
        sanitize(dealer)
        for dealer in scored_dealers
        if dealer["dealer_id"] != best_dealer["dealer_id"]
        and dealer["distance_km"] <= ALTERNATIVE_DISTANCE_LIMIT_KM
    ][:3]

    return {
        "best_dealer": best_dealer,
        "alternative_dealers": alternatives,
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
    Automatically recommends and assigns the best dealer using the allocation model
    Returns alternative dealer options
    """
    
    try:
        # Step 1: Geocode the lead address
        try:
            lead_lat, lead_lon = geocode_address(lead.address, lead.city, lead.province)
        except HTTPException as he:
            # If geocoding fails, still create the lead but don't assign a dealer
            lead_lat, lead_lon = None, None
        
        # Step 2: Get allocation (best + alternatives) if geocoding succeeded
        allocation = None
        if lead_lat and lead_lon:
            allocation = allocate_lead_to_dealer(lead_lat, lead_lon, lead.province, lead)
        
        # Step 3: Insert lead into database
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                best_dealer = allocation["best_dealer"] if allocation else None
                assigned_dealer_id = best_dealer["dealer_id"] if best_dealer else None
                recommended_dealer_id = best_dealer["dealer_id"] if best_dealer else None
                final_dealer_selection = best_dealer["dealer_name"] if best_dealer else None
                query = """
                    INSERT INTO leads (
                        name, email, phone, address, city, province, postal_code,
                        job_type, comments, status, assigned_dealer_id, recommended_dealer_id,
                        allocation_score, distance_to_dealer_km, dealer_ml_probability,
                        final_dealer_selection,
                        latitude, longitude, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
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
                    assigned_dealer_id,
                    recommended_dealer_id,
                    best_dealer['allocation_score'] if best_dealer else None,
                    best_dealer['distance_km'] if best_dealer else None,
                    best_dealer.get('ml_probability') if best_dealer else None,
                    final_dealer_selection,
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
        
        # Step 4: Format alternative dealers response
        alternative_dealers = None
        if allocation and allocation['alternative_dealers']:
            alternative_dealers = [
                AlternativeDealer(
                    id=alt['dealer_id'],
                    name=alt['dealer_name'],
                    city=alt['city'],
                    province=alt['province'],
                    distance_km=alt['distance_km'],
                    allocation_score=alt['allocation_score'],
                    active_leads=alt['active_leads'],
                    converted_leads=alt.get('converted_leads'),
                    ml_probability=alt.get('ml_probability'),
                    distance_review_required=alt.get('distance_review_required'),
                ) for alt in allocation['alternative_dealers']
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
            assigned_dealer_id=best_dealer['dealer_id'] if allocation else None,
            dealer_name_assigned=best_dealer['dealer_name'] if allocation else None,
            recommended_dealer_id=best_dealer['dealer_id'] if allocation else None,
            recommended_dealer_name=best_dealer['dealer_name'] if allocation else None,
            final_dealer_selection=best_dealer['dealer_name'] if allocation else None,
            allocation_score=best_dealer['allocation_score'] if allocation else None,
            distance_to_dealer_km=best_dealer['distance_km'] if allocation else None,
            dealer_ml_probability=best_dealer.get('ml_probability') if allocation else None,
            distance_review_required=best_dealer.get('distance_review_required') if allocation else None,
            alternative_dealers=alternative_dealers,
            created_at=created_at,
            message="Lead submitted and dealer assigned successfully" if allocation else "Lead submitted - no dealer available in your area"
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
            COUNT(CASE WHEN assigned_dealer_id IS NOT NULL THEN 1 END) as assigned_leads,
            COUNT(CASE WHEN status = 'converted' THEN 1 END) as completed_leads,
            AVG(allocation_score) as avg_allocation_score,
            (COUNT(CASE WHEN status = 'converted' THEN 1 END)::float / 
             NULLIF(COUNT(CASE WHEN status IN ('converted', 'dead') THEN 1 END), 0) * 100) as conversion_rate
        FROM leads
    """
    
    stats = execute_query(query)[0]
    
    active_dealers = execute_query("SELECT COUNT(*) as count FROM dealers WHERE is_active = TRUE")[0]['count']
    
    return DashboardStats(
        total_leads=stats['total_leads'] or 0,
        pending_leads=stats['pending_leads'] or 0,
        assigned_leads=stats['assigned_leads'] or 0,
        completed_leads=stats['completed_leads'] or 0,
        conversion_rate=round(stats['conversion_rate'] or 0, 2),
        avg_allocation_score=round(stats['avg_allocation_score'] or 0, 2),
        active_dealers=active_dealers
    )

@app.get("/api/admin/leads")
async def get_all_leads(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: AdminUser = Depends(get_current_user)
):
    """Get all leads with optional status filter - includes alternative dealers"""
    
    if status:
        query = """
            SELECT l.*, d.name as dealer_name_assigned, rd.name as recommended_dealer_name
            FROM leads l
            LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
            LEFT JOIN dealers rd ON l.recommended_dealer_id = rd.id
            WHERE l.status = %s
            ORDER BY l.created_at DESC
            LIMIT %s OFFSET %s
        """
        params = (status, limit, offset)
    else:
        query = """
            SELECT l.*, d.name as dealer_name_assigned, rd.name as recommended_dealer_name
            FROM leads l
            LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
            LEFT JOIN dealers rd ON l.recommended_dealer_id = rd.id
            ORDER BY l.created_at DESC
            LIMIT %s OFFSET %s
        """
        params = (limit, offset)
    
    leads = execute_query(query, params)
    
    # Cache dealer pools per province within this request to avoid repetitive queries
    dealer_cache: Dict[str, List[Dict[str, Any]]] = {}
    try:
        historical_stats = fetch_dealer_historical_feature_stats(execute_query)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Unable to preload historical dealer stats: %s", exc)
        historical_stats = None

    # For each lead, calculate alternative dealers if coordinates exist
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
            dealer_pool: Optional[List[Dict[str, Any]]] = None
            if province_key:
                if province_key not in dealer_cache:
                    try:
                        dealer_cache[province_key] = fetch_active_dealers_by_province(province_key)
                    except HTTPException as exc:
                        logger.warning(
                            "Dealer lookup failed for province %s: %s",
                            province_key,
                            getattr(exc, 'detail', str(exc)),
                        )
                        dealer_cache[province_key] = []
                dealer_pool = dealer_cache.get(province_key)
            try:
                allocation = allocate_lead_to_dealer(
                    lead_lat,
                    lead_lon,
                    lead.get('province'),
                    lead,
                    dealer_pool=dealer_pool,
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
            lead_dict['recommended_dealer_id'] = allocation['best_dealer'].get('dealer_id')
            lead_dict['recommended_dealer_name'] = allocation['best_dealer'].get('dealer_name')
            lead_dict['dealer_ml_probability'] = allocation['best_dealer'].get('ml_probability')
            lead_dict['distance_review_required'] = allocation['best_dealer'].get('distance_review_required')
            lead_dict['distance_to_dealer_km'] = allocation['best_dealer'].get('distance_km')
            lead_dict['alternative_dealers'] = allocation['alternative_dealers']
        else:
            lead_dict['alternative_dealers'] = []
            lead_dict['distance_review_required'] = None

        resolved_final = resolve_final_dealer_selection(lead_dict)
        if resolved_final:
            lead_dict['final_dealer_selection'] = resolved_final

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
    """Get detailed information about a specific lead - includes alternative dealers"""
    
    query = """
        SELECT l.*, 
               d.name as dealer_name_assigned,
               d.email as dealer_email_assigned,
               d.phone as dealer_phone_assigned,
               d.city as dealer_city_assigned,
               rd.name as recommended_dealer_name,
               rd.email as recommended_dealer_email,
               rd.phone as recommended_dealer_phone,
               rd.city as recommended_dealer_city
        FROM leads l
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        LEFT JOIN dealers rd ON l.recommended_dealer_id = rd.id
        WHERE l.id = %s
    """
    
    lead = execute_query(query, (lead_id,))
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    lead_dict = dict(lead[0])
    
    allocation = None
    if lead_dict['latitude'] and lead_dict['longitude']:
        try:
            allocation = allocate_lead_to_dealer(
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
        lead_dict['recommended_dealer_id'] = allocation['best_dealer'].get('dealer_id')
        lead_dict['recommended_dealer_name'] = allocation['best_dealer'].get('dealer_name')
        lead_dict['dealer_ml_probability'] = allocation['best_dealer'].get('ml_probability')
        lead_dict['distance_review_required'] = allocation['best_dealer'].get('distance_review_required')
        lead_dict['distance_to_dealer_km'] = allocation['best_dealer'].get('distance_km')
        lead_dict['alternative_dealers'] = allocation['alternative_dealers']
    else:
        lead_dict['alternative_dealers'] = []
        lead_dict['distance_review_required'] = None

    resolved_final = resolve_final_dealer_selection(lead_dict)
    if resolved_final:
        lead_dict['final_dealer_selection'] = resolved_final

    return lead_dict

@app.patch("/api/admin/leads/{lead_id}/status")
async def update_lead_status(
    lead_id: int,
    status: str,
    current_user: AdminUser = Depends(get_current_user)
):
    """Update lead status.

    The historical sync expects *any* non-active status change to be persisted, so we
    intentionally avoid over-validating allowed status values here. Frontend clients
    still send canonical options (converted, dead, follow_up), but custom labels such
    as "Converted Sale" must continue to flow through so the historical table is
    populated.
    """
    
    lead_rows = execute_query(
        """
        SELECT l.*, d.name as dealer_name_assigned
        FROM leads l
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        WHERE l.id = %s
        """,
        (lead_id,),
    )

    if not lead_rows:
        raise HTTPException(status_code=404, detail="Lead not found")

    query = "UPDATE leads SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
    execute_query(query, (status, lead_id), fetch=False)

    # Re-read the lead so historical sync uses the persisted state and captures
    # downstream changes to the final dealer selection.
    sync_lead_to_historical(lead_id)

    return {"message": "Lead status updated successfully", "lead_id": lead_id, "new_status": status}


@app.get("/api/admin/historical-data")
async def get_historical_data(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    current_user: AdminUser = Depends(get_current_user)
):
    """Get Lasso-backed historical data records."""

    conditions = ["1=1"]
    params: list[Any] = []
    if status and status != 'all':
        conditions.append("status_bucket = %s")
        params.append(status)

    where_clause = " AND ".join(conditions)
    data = execute_query(
        f"""
            SELECT
                lasso_lead_id AS id,
                COALESCE(submit_date, created_date, form_submit_date) AS submit_date,
                first_name,
                last_name,
                company_name,
                address AS address1,
                city,
                province,
                postal_code AS postal,
                dealer_name,
                project_type,
                product_type,
                square_footage_value AS square_footage,
                current_status,
                value_of_order,
                synced_at AS created_at
            FROM dashboard_history_leads
            WHERE {where_clause}
            ORDER BY last_interaction DESC NULLS LAST, created_date DESC NULLS LAST
            LIMIT %s OFFSET %s
        """,
        tuple([*params, limit, offset]),
    )

    total_count = execute_query(
        f"SELECT COUNT(*) as total FROM dashboard_history_leads WHERE {where_clause}",
        tuple(params) if params else None,
    )[0]['total']
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
# MIGRATION RUNNER
# ============================================
@app.on_event("startup")
async def run_migrations():
    """Run SQL migrations on startup."""
    import glob
    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        return
    migration_files = sorted(glob.glob(str(migrations_dir / "*.sql")))
    for mf in migration_files:
        conn = None
        try:
            with open(mf, 'r') as f:
                sql = f.read()
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql)
            conn.commit()
            logger.info(f"Migration applied: {Path(mf).name}")
        except Exception as e:
            if conn is not None and getattr(conn, "closed", 1) == 0:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.warning(f"Migration {Path(mf).name} skipped or failed: {e}")
        finally:
            if conn is not None and getattr(conn, "closed", 1) == 0:
                conn.close()

# ============================================
# BACKGROUND SYNC SCHEDULER
# ============================================

import threading as _threading

def _lasso_scheduler_loop() -> None:
    """
    Background thread that drives Lasso syncs on a schedule.
    Fast sync every lasso_fast_sync_interval_minutes (default 5 min).
    Full sync every lasso_full_sync_interval_minutes (default 15 min).
    Pages never trigger syncs themselves — all syncs originate here or
    from an explicit admin action (Tools button / POST /sync).
    """
    import time as _time
    from services.lasso_dashboard_sync import (
        start_lasso_dashboard_sync,
        _last_successful_sync,
        _current_running_sync,
        SYNC_STATUS,
    )
    from settings_store import get_setting_float
    from datetime import datetime, timedelta

    _time.sleep(10)  # let the app finish starting up before first check

    while True:
        try:
            if not SYNC_STATUS["in_progress"] and not _current_running_sync():
                fast_interval = int(round(get_setting_float("lasso_fast_sync_interval_minutes", 5.0)))
                full_interval = int(round(get_setting_float("lasso_full_sync_interval_minutes", 15.0)))
                last = _last_successful_sync()
                now = datetime.utcnow()

                if last is None:
                    start_lasso_dashboard_sync(sync_type="full", actor="scheduler")
                elif last.get("completed_at"):
                    age_minutes = (now - last["completed_at"]).total_seconds() / 60
                    # Check if a full sync is due first
                    last_full_rows = None
                    try:
                        from db import execute_query as _eq
                        rows = _eq(
                            "SELECT completed_at FROM lasso_sync_runs "
                            "WHERE status='completed' AND sync_type='full' "
                            "ORDER BY completed_at DESC NULLS LAST LIMIT 1"
                        )
                        last_full_rows = rows[0] if rows else None
                    except Exception:
                        pass
                    full_age = None
                    if last_full_rows and last_full_rows.get("completed_at"):
                        full_age = (now - last_full_rows["completed_at"]).total_seconds() / 60
                    if full_age is None or full_age >= full_interval:
                        start_lasso_dashboard_sync(sync_type="full", actor="scheduler")
                    elif age_minutes >= fast_interval:
                        start_lasso_dashboard_sync(sync_type="fast", actor="scheduler")
        except Exception as exc:
            logger.warning(f"Lasso scheduler error: {exc}")

        _time.sleep(60)  # check every 60 s; the interval logic above decides whether to act


@app.on_event("startup")
async def start_lasso_scheduler():
    t = _threading.Thread(target=_lasso_scheduler_loop, name="lasso-scheduler", daemon=True)
    t.start()
    logger.info("Lasso background sync scheduler started")


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
