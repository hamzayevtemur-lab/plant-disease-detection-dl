"""
Top-level entry point for cloud deployments (Render, Railway, Hugging Face, Heroku).
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting FloraGuard AI on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
