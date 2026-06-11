"""
Start the Crop Yield Prediction API Server
"""

import uvicorn
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("="*60)
    print("🌾 Starting Crop Yield Prediction API Server")
    print("="*60)
    print("\n📁 Loading trained models from 'models' directory...")
    print("API will be available at: http://127.0.0.1:8000")
    print("Documentation: http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*60)
    
    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )