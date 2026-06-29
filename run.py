"""Entry point for the DualGAT Stock Prediction system."""
import uvicorn
from config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "src.web.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
