"""
Vercel serverless entry point. Exposes the same FastAPI app as app/main.py so
`vercel deploy` serves the whole demo (page + /api/query/*) from one function.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402
