"""
FastAPI Application - Energy Demand ML API
Author: Emad Malik
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import sys
from pathlib import Path
import subprocess
import json

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import EnergyDemandPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Energy Demand ML API",
    description="API for energy demand prediction using XGBoost",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")), name="static")

# Global predictor instance
predictor: Optional[EnergyDemandPredictor] = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    demand_forecast_mw: float = Field(..., description="Forecasted demand in MW", example=5000.0)
    net_generation_mw: float = Field(..., description="Net generation in MW", example=4800.0)
    total_interchange_mw: float = Field(..., description="Total interchange in MW", example=200.0)
    hour_number: int = Field(..., ge=1, le=25, description="Hour number (1-25)", example=14)
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)", example=14)
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)", example=2)
    month: int = Field(..., ge=1, le=12, description="Month (1-12)", example=7)
    balancing_authority: str = Field(..., description="Balancing Authority", example="CISO")
    sub_region: str = Field(..., description="Sub-Region", example="PGAE")
    season: str = Field(..., description="Season", example="summer")

    class Config:
        json_schema_extra = {
            "example": {
                "demand_forecast_mw": 5000.0,
                "net_generation_mw": 4800.0,
                "total_interchange_mw": 200.0,
                "hour_number": 14,
                "hour": 14,
                "day_of_week": 2,
                "month": 7,
                "balancing_authority": "CISO",
                "sub_region": "PGAE",
                "season": "summer"
            }
        }

class PredictionResponse(BaseModel):
    predicted_demand_mw: float
    lower_bound_mw: float
    upper_bound_mw: float
    model_test_mae: float
    model_test_rmse: float
    model_test_r2: float

class TrainingStatus(BaseModel):
    status: str
    message: str

# Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global predictor
    try:
        predictor = EnergyDemandPredictor()
        print("✓ Model loaded successfully on startup")
    except FileNotFoundError:
        print("⚠ Warning: Model not found. Please train the model first.")
        predictor = None

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - redirect to dashboard"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Energy Demand ML API"
    })

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        return {
            "status": "unhealthy",
            "message": "Model not loaded. Please train the model first.",
            "model_loaded": False
        }
    
    return {
        "status": "healthy",
        "message": "API is running and model is loaded",
        "model_loaded": True,
        "model_metrics": predictor.metrics
    }

# Endpoint 1: Training
@app.post("/train", response_model=TrainingStatus)
async def train_model(background_tasks: BackgroundTasks):
    """
    Trigger model training
    This runs the training pipeline in the background
    """
    def run_training():
        """Background task to run training"""
        try:
            # Run training script
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "train.py")],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            if result.returncode == 0:
                # Reload the predictor with new model
                global predictor
                predictor = EnergyDemandPredictor()
                print("✓ Model retrained and reloaded successfully")
            else:
                print(f"✗ Training failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Training error: {str(e)}")
    
    # Add training to background tasks
    background_tasks.add_task(run_training)
    
    return TrainingStatus(
        status="started",
        message="Model training started in background. Check /health for status."
    )

# Endpoint 2: Prediction
@app.post("/predict/xgboost", response_model=PredictionResponse)
async def predict_xgboost(request: PredictionRequest):
    """
    Make a prediction using the XGBoost model
    
    Returns predicted demand with confidence interval
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using POST /train"
        )
    
    try:
        # Prepare input data
        input_data = {
            'Demand Forecast (MW)': request.demand_forecast_mw,
            'Net Generation (MW)': request.net_generation_mw,
            'Total Interchange (MW)': request.total_interchange_mw,
            'Hour Number': request.hour_number,
            'hour': request.hour,
            'day_of_week': request.day_of_week,
            'month': request.month,
            'Balancing Authority': request.balancing_authority,
            'Sub-Region': request.sub_region,
            'season': request.season
        }
        
        # Make prediction
        result = predictor.predict_with_confidence(input_data)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Endpoint 3: Dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Serve the dashboard HTML page
    """
    # Get model status and metrics
    model_status = {
        "loaded": predictor is not None,
        "metrics": predictor.metrics if predictor else None
    }
    
    # Example predictions for display
    example_data = {
        "demand_forecast_mw": 5000.0,
        "net_generation_mw": 4800.0,
        "total_interchange_mw": 200.0,
        "hour_number": 14,
        "hour": 14,
        "day_of_week": 2,
        "month": 7,
        "balancing_authority": "CISO",
        "sub_region": "PGAE",
        "season": "summer"
    }
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Energy Demand Prediction Dashboard",
        "model_status": model_status,
        "example_data": example_data
    })

# API Info endpoint
@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "api_name": "Energy Demand ML API",
        "version": "2.1.0",
        "message": "Testing CI/CD pipeline",
        "endpoints": {
            "/": "Home page",
            "/health": "Health check and model status",
            "/train": "POST - Trigger model training",
            "/predict/xgboost": "POST - Make prediction",
            "/dashboard": "Interactive dashboard",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)"
        },
        "model_loaded": predictor is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
