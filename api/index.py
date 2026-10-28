"""
api/index.py - Vercel Serverless Function Entrypoint
Exposes the FestNest AutoFill Flask application for Vercel Fluid Compute / AWS Lambda.
"""
import sys
from pathlib import Path

# Add project root to sys.path so modules and packages (app, scripts) can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

# Expose WSGI application instance for Vercel
app = app
handler = app
