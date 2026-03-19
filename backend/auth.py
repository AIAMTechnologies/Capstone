import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from db import settings, execute_query

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/admin/login")


class AdminUser(BaseModel):
    id: int
    username: str
    email: str
    last_name: Optional[str] = None
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: AdminUser


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def resolve_admin_user_from_token(token: str) -> Optional[AdminUser]:
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
    return bool(
        provided_key
        and settings.ML_PUBLIC_API_KEY
        and secrets.compare_digest(provided_key, settings.ML_PUBLIC_API_KEY)
    )
