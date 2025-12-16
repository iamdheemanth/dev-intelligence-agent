# app/auth.py
import os
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN", "")

bearer = HTTPBearer(auto_error=False)

def require_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not API_TOKEN:
        return 
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = credentials.credentials
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
