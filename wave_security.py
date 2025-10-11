import os
import secrets
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

def get_current_username(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    """Validates username and password for secured routes."""
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: missing admin credentials."
        )

    username_bytes = credentials.username.encode("utf-8")
    password_bytes = credentials.password.encode("utf-8")

    if not (
        secrets.compare_digest(username_bytes, ADMIN_USERNAME.encode("utf-8"))
        and secrets.compare_digest(password_bytes, ADMIN_PASSWORD.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username