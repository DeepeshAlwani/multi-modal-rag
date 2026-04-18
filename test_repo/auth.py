"""
User authentication utilities.
"""

def verify_user(token: str) -> bool:
    """Mock token verification."""
    return token == "valid_token_123"

def get_user_role(token: str) -> str:
    """Return role for a valid token."""
    if verify_user(token):
        return "admin"
    return "guest"