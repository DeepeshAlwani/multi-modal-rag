"""
conftest.py — shared pytest fixtures.

Run from project root:
    pytest tests/ -v
"""
import sys
import os
import pytest

# Make sure project root is importable (so 'from query_engine import ...' works)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_functions():
    """A minimal list of (doc, metadata) pairs mimicking ChromaDB output."""
    return [
        ("def verify_user(email, password): ...", {"function": "verify_user",  "file": "database.py", "lines": "(10, 30)"}),
        ("def create_session(user_id): ...",      {"function": "create_session","file": "database.py", "lines": "(32, 50)"}),
        ("def check_rate_limit(...): ...",        {"function": "check_rate_limit","file": "database.py","lines": "(52, 80)"}),
    ]


@pytest.fixture
def sample_diagram():
    return [
        ("Payment flow: user → gateway → bank", {"file": "payment.png", "type": "diagram"}),
    ]