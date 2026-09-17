"""
Top-level entry point for cloud deployments (Hugging Face Spaces, Render, Railway, Heroku).
"""

import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 is default for Hugging Face Spaces, 8000/10000 for others
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
