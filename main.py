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
from typing import List, Optional
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

# ============================================
# CONFIGURATION
# ============================================

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:4567@localhost:5432/capstone25")
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
    created_at: datetime
    message: str = "Lead submitted successfully"

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
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=408, detail="Geocoding service timed out. Please try again.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Geocoding error: {str(e)}")

# ============================================
# LEAD ALLOCATION ALGORITHM
# ============================================

def calculate_allocation_score(
    lead_lat: float,
    lead_lon: float,
    lead_job_type: str,
    installer: dict
) -> dict:
    """
    Calculate allocation score for a lead-installer pair
    
    Scoring factors:
    1. Distance (50% weight) - closer is better
    2. Capacity (30% weight) - less busy is better
    3. Success Rate (20% weight) - higher success rate is better
    
    Additional factors:
    - Job type match (bonus for specialization)
    - Travel time estimate
    
    Returns dict with score and metadata
    """
    
    # 1. Calculate distance
    distance_km = haversine_distance(
        lead_lat, lead_lon,
        float(installer['latitude']), float(installer['longitude'])
    )
    
    # Distance score (0-100 scale)
    # Normalized against service radius
    service_radius = installer.get('service_radius_km', 100)
    if distance_km > service_radius:
        distance_score = 100  # Outside service area - maximum penalty
    else:
        distance_score = (distance_km / service_radius) * 50
    
    # 2. Capacity score (0-50 scale)
    max_capacity = installer.get('max_capacity', 10)
    current_capacity = installer.get('current_capacity', 0)
    
    if current_capacity >= max_capacity:
        return None  # Installer at full capacity, skip
    
    capacity_ratio = current_capacity / max_capacity
    capacity_score = capacity_ratio * 50
    
    # 3. Success rate score (0-30 scale, inverted)
    success_rate = float(installer.get('success_rate', 0.7))
    success_score = (1 - success_rate) * 30
    
    # 4. Job type match bonus
    specialization = installer.get('specialization', [])
    job_type_bonus = 0
    if isinstance(specialization, list) and lead_job_type in specialization:
        job_type_bonus = -5  # Reduce score by 5 points (bonus)
    
    # 5. Calculate total score
    total_score = (
        distance_score * 0.5 +      # 50% weight
        capacity_score * 0.3 +       # 30% weight
        success_score * 0.2 +        # 20% weight
        job_type_bonus               # Bonus adjustment
    )
    
    # 6. Estimate travel time (assuming 60 km/h average speed)
    travel_time_minutes = (distance_km / 60) * 60
    
    return {
        'installer_id': installer['id'],
        'installer_name': installer['name'],
        'allocation_score': round(total_score, 2),
        'distance_km': round(distance_km, 2),
        'travel_time_minutes': round(travel_time_minutes, 2),
        'current_capacity': current_capacity,
        'max_capacity': max_capacity,
        'success_rate': success_rate,
        'specialization_match': job_type_bonus < 0
    }

def allocate_lead_to_installer(
    lead_lat: float,
    lead_lon: float,
    lead_province: str,
    lead_job_type: str
) -> Optional[dict]:
    """
    Find the best installer for a lead based on multiple factors
    Returns allocation details or None if no suitable installer found
    """
    
    # Fetch eligible installers from database
    query = """
        SELECT id, name, city, province, latitude, longitude,
               service_radius_km, max_capacity, current_capacity,
               success_rate, specialization, is_active
        FROM installers
        WHERE is_active = TRUE
          AND province = %s
          AND current_capacity < max_capacity
        ORDER BY current_capacity ASC
    """
    
    installers = execute_query(query, (lead_province,))
    
    if not installers:
        # No installers available in this province
        return None
    
    # Score each installer
    scored_installers = []
    for installer in installers:
        score_result = calculate_allocation_score(
            lead_lat, lead_lon, lead_job_type, installer
        )
        if score_result:  # Only include if not at capacity
            scored_installers.append(score_result)
    
    if not scored_installers:
        return None
    
    # Sort by allocation score (lower is better)
    scored_installers.sort(key=lambda x: x['allocation_score'])
    
    # Return best match
    return scored_installers[0]

# ============================================
# AUTHENTICATION
# ============================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> AdminUser:
    """Get current authenticated user from JWT token"""
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
    """API health check"""
    return {
        "status": "online",
        "message": "Lead Allocation System API",
        "version": "1.0.0"
    }

@app.post("/api/leads", response_model=LeadResponse, status_code=status.HTTP_201_CREATED)
async def submit_lead(lead: LeadCreate):
    """
    PUBLIC ENDPOINT: Submit a new lead from the client form
    
    Process:
    1. Validate input data
    2. Geocode address to coordinates
    3. Find best installer using allocation algorithm
    4. Store lead in database with assignment
    5. Return lead details with assigned installer
    """
    
    try:
        # Step 1: Geocode address
        latitude, longitude = geocode_address(lead.address, lead.city, lead.province)
        
        # Step 2: Find best installer
        allocation = allocate_lead_to_installer(
            latitude, longitude, lead.province, lead.job_type
        )
        
        # Step 3: Insert lead into database
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                if allocation:
                    insert_query = """
                        INSERT INTO leads (
                            name, email, phone, address, city, province, postal_code,
                            latitude, longitude, job_type, comments,
                            assigned_installer_id, allocation_score, 
                            distance_to_installer_km, estimated_travel_time_minutes,
                            status, source, assigned_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                        ) RETURNING id, created_at
                    """
                    params = (
                        lead.name, lead.email, lead.phone, lead.address, lead.city,
                        lead.province, lead.postal_code, latitude, longitude,
                        lead.job_type, lead.comments, allocation['installer_id'], 
                        allocation['allocation_score'], allocation['distance_km'], 
                        allocation['travel_time_minutes'], 'active', 'website'
                    )
                else:
                    insert_query = """
                        INSERT INTO leads (
                            name, email, phone, address, city, province, postal_code,
                            latitude, longitude, job_type, comments,
                            status, source
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) RETURNING id, created_at
                    """
                    params = (
                        lead.name, lead.email, lead.phone, lead.address, lead.city,
                        lead.province, lead.postal_code, latitude, longitude,
                        lead.job_type, lead.comments, 'active', 'website'
                    )
                
                cursor.execute(insert_query, params)
                result = cursor.fetchone()
                lead_id = result['id']
                created_at = result['created_at']
                conn.commit()  # COMMIT THE TRANSACTION
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to insert lead: {str(e)}")
        finally:
            conn.close()
        
        # Step 4: Return response
        return LeadResponse(
            id=lead_id,
            name=lead.name,
            email=lead.email,
            city=lead.city,
            province=lead.province,
            job_type=lead.job_type,
            status='active',
            assigned_installer_id=allocation['installer_id'] if allocation else None,
            assigned_installer_name=allocation['installer_name'] if allocation else None,
            allocation_score=allocation['allocation_score'] if allocation else None,
            distance_to_installer_km=allocation['distance_km'] if allocation else None,
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
    """Get all leads with optional status filter"""
    
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
    
    # Also get total count
    count_query = "SELECT COUNT(*) as total FROM leads"
    if status:
        count_query += " WHERE status = %s"
        total_count = execute_query(count_query, (status,))[0]['total']
    else:
        total_count = execute_query(count_query)[0]['total']
    
    return {"leads": leads, "count": len(leads), "total": total_count}

@app.get("/api/admin/leads/{lead_id}")
async def get_lead_detail(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Get detailed information about a specific lead"""
    
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
    
    return lead[0]

@app.patch("/api/admin/leads/{lead_id}/status")
async def update_lead_status(
    lead_id: int,
    status: str,
    current_user: AdminUser = Depends(get_current_user)
):
    """Update lead status"""
    
    valid_statuses = ['active', 'converted', 'dead']
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
    
    query = "UPDATE leads SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
    execute_query(query, (status, lead_id), fetch=False)
    
    return {"message": "Lead status updated successfully", "lead_id": lead_id, "new_status": status}

@app.get("/api/admin/installers")
async def get_installers(current_user: AdminUser = Depends(get_current_user)):
    """Get all installers with performance metrics"""
    
    query = """
        SELECT * FROM installer_performance
        ORDER BY province, city
    """
    
    installers = execute_query(query)
    return {"installers": installers, "count": len(installers)}

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)