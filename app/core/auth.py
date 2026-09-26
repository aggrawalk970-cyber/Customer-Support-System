from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key_header: str = Security(api_key_header)):
    if not settings.API_KEYS:
        # If no keys are configured, skip authentication (dev mode)
        return True
        
    valid_keys = [k.strip() for k in settings.API_KEYS.split(",") if k.strip()]
    
    if not valid_keys:
        return True

    if api_key_header in valid_keys:
        return True
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-Key header",
    )
