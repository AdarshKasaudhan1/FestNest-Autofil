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

# WSGI Middleware to ensure PATH_INFO is cleanly routed across Vercel rewrite patterns
class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        # Normalize /api/index to / or stripped path
        if path == "/api/index" or path == "/api/index/":
            environ["PATH_INFO"] = "/"
        elif path.startswith("/api/index/"):
            environ["PATH_INFO"] = path[len("/api/index"):]
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFix(app.wsgi_app)

# Expose WSGI application instance for Vercel
handler = app
