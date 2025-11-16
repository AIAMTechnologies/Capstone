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

import os
import time
from datetime import datetime, timedelta
from typing import Any, List, Optional
from math import radians, sin, cos, sqrt, atan2

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from jose import JWTError, jwt
from passlib.context import CryptContext
import psycopg2
from psycopg2.extras import RealDictCursor
import requests

from ml_model import InstallerMLModel

# ============================================
# CONFIGURATION
# ============================================

class Settings:
    #DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:4567@localhost:5432/capstone25")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://q4gems_admin:890*()iopIOP@capstone25.postgres.database.azure.com:5432/capstone25db")  #Azure specific
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours
    
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


# Initialize the ML allocator once so it can be reused across requests
ml_allocator = InstallerMLModel(execute_query)

# Distance preferences used when evaluating installer suitability
PREFERRED_DISTANCE_KM = 120
ALTERNATIVE_DISTANCE_LIMIT_KM = 50
MAX_DISTANCE_KM = 400

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
        "product_type": payload.get("product_type"),
        "square_footage": payload.get("square_footage") or payload.get("square_feet"),
        "current_status": payload.get("current_status") or payload.get("status"),
    }

# ============================================
# LEAD ALLOCATION ALGORITHM - ENHANCED VERSION
# ============================================

def allocate_lead_to_installer(
    lead_lat: float,
    lead_lon: float,
    province: str,
    lead_payload: Optional[Any] = None,
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
    
    if not installers:
        return None
    
    # Prepare ML probabilities for this lead
    lead_features = build_ml_feature_payload(lead_payload)
    ml_probabilities = ml_allocator.predict_probabilities(lead_features)

    # Calculate scores for all installers
    scored_installers = []
    max_closed = max((installer.get('converted_leads') or 0) for installer in installers) if installers else 0
    max_active = max((installer.get('active_leads') or 0) for installer in installers) if installers else 0

    for installer in installers:
        # Calculate distance
        distance_km = haversine_distance(
            lead_lat, lead_lon,
            installer['latitude'], installer['longitude']
        )

        if distance_km > MAX_DISTANCE_KM:
            # Skip installers that are far outside the supported region
            continue

        installer_name = installer['name']
        probability = (
            ml_probabilities.get(installer_name)
            or ml_probabilities.get(installer_name.lower())
            or 0.0
        )
        closed_leads = installer.get('converted_leads') or 0
        active_leads = installer.get('active_leads') or 0

        distance_component = max(0.0, 1 - min(distance_km / 300, 1))
        closed_component = (closed_leads / max_closed) if max_closed else 0
        active_component = (active_leads / max_active) if max_active else 0

        allocation_score = (
            (probability * 0.6)
            + (distance_component * 0.2)
            + (closed_component * 0.15)
            - (active_component * 0.15)
        )

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
            'distance_review_required': False,
        })

    # Sort by allocation score (higher scores are better)
    scored_installers.sort(key=lambda x: x['allocation_score'], reverse=True)

    if not scored_installers:
        return None

    # Prefer installers that are within the desired distance threshold
    preferred_installers = [
        installer for installer in scored_installers
        if installer['distance_km'] <= PREFERRED_DISTANCE_KM
    ]

    if preferred_installers:
        best_installer = preferred_installers[0]
    else:
        best_installer = scored_installers[0]
        best_installer['distance_review_required'] = True

    # Get alternative installers within 50km (excluding the best one)
    alternatives = [
        installer for installer in scored_installers
        if installer['installer_id'] != best_installer['installer_id']
        and installer['distance_km'] <= ALTERNATIVE_DISTANCE_LIMIT_KM
    ][:3]  # Limit to 3 alternatives
    
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

async def get_current_user(token: str = Depends(oauth2_scheme)) -> AdminUser:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    query = "SELECT id, username, email, last_name, role FROM admin_users WHERE username = %s AND is_active = TRUE"
    user = execute_query(query, (username,))
    
    if not user:
        raise credentials_exception
    
    return AdminUser(**user[0])

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
                query = """
                    INSERT INTO leads (
                        name, email, phone, address, city, province, postal_code,
                        job_type, comments, status, assigned_installer_id, 
                        allocation_score, distance_to_installer_km,
                        latitude, longitude, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
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
                    allocation['best_installer']['installer_id'] if allocation else None,
                    allocation['best_installer']['allocation_score'] if allocation else None,
                    allocation['best_installer']['distance_km'] if allocation else None,
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
    
    # For each lead, calculate alternative installers if coordinates exist
    enhanced_leads = []
    for lead in leads:
        lead_dict = dict(lead)
        
        # Calculate alternative installers if lead has coordinates
        if lead['latitude'] and lead['longitude']:
            allocation = allocate_lead_to_installer(
                lead['latitude'],
                lead['longitude'],
                lead['province'],
                lead
            )

            if allocation:
                lead_dict['installer_ml_probability'] = allocation['best_installer'].get('ml_probability')
                lead_dict['distance_review_required'] = allocation['best_installer'].get('distance_review_required')
                lead_dict['alternative_installers'] = allocation['alternative_installers']
            else:
                lead_dict['alternative_installers'] = []
                lead_dict['distance_review_required'] = None
        else:
            lead_dict['alternative_installers'] = []
            lead_dict['distance_review_required'] = None
        
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
    
    # Calculate alternative installers if lead has coordinates
    if lead_dict['latitude'] and lead_dict['longitude']:
        allocation = allocate_lead_to_installer(
            lead_dict['latitude'],
            lead_dict['longitude'],
            lead_dict['province'],
            lead_dict
        )

        if allocation:
            lead_dict['installer_ml_probability'] = allocation['best_installer'].get('ml_probability')
            lead_dict['distance_review_required'] = allocation['best_installer'].get('distance_review_required')
            lead_dict['alternative_installers'] = allocation['alternative_installers']
        else:
            lead_dict['alternative_installers'] = []
            lead_dict['distance_review_required'] = None
    else:
        lead_dict['alternative_installers'] = []
        lead_dict['distance_review_required'] = None
    
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
    
    query = "UPDATE leads SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
    execute_query(query, (status, lead_id), fetch=False)
    
    return {"message": "Lead status updated successfully", "lead_id": lead_id, "new_status": status}

@app.patch("/api/admin/leads/{lead_id}/installer-override")
async def update_installer_override(
    lead_id: int,
    installer_id: Optional[int] = None,
    current_user: AdminUser = Depends(get_current_user)
):
    """Update installer override - allows manual assignment of alternative installers"""
    
    # Validate installer exists if provided
    if installer_id:
        installer_check = execute_query(
            "SELECT id FROM installers WHERE id = %s AND is_active = TRUE",
            (installer_id,)
        )
        if not installer_check:
            raise HTTPException(status_code=404, detail="Installer not found or inactive")
    
    query = """
        UPDATE leads 
        SET installer_override_id = %s, updated_at = CURRENT_TIMESTAMP 
        WHERE id = %s
    """
    execute_query(query, (installer_id, lead_id), fetch=False)
    
    return {
        "message": "Installer override updated successfully", 
        "lead_id": lead_id, 
        "installer_id": installer_id
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


@app.get("/api/admin/ml/status", response_model=MLStatusResponse)
async def get_ml_status(current_user: AdminUser = Depends(get_current_user)):
    """Return the current ML training status for administrators."""

    status = ml_allocator.status()
    return MLStatusResponse(**status, message="ok" if status.get("trained") else "not_trained")


@app.post("/api/admin/ml/train", response_model=MLStatusResponse)
async def trigger_ml_training(
    payload: Optional[MLTrainRequest] = None,
    current_user: AdminUser = Depends(get_current_user)
):
    """Allow administrators to retrain the ML model on demand."""

    payload = payload or MLTrainRequest()
    success = ml_allocator.train(force=payload.force)
    status = ml_allocator.status()
    message = "trained" if success else "training_failed"
    return MLStatusResponse(**status, message=message)

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)