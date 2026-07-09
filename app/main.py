"""
FastAPI Application - Energy Demand ML API
Author: Emad Malik
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import os
import sys
import time
from pathlib import Path
import subprocess
import json

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import EnergyDemandPredictor

# Global predictor instance
predictor: Optional[EnergyDemandPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup (lifespan replaces the deprecated
    @app.on_event('startup') hook, which newer FastAPI versions remove)."""
    global predictor
    try:
        predictor = EnergyDemandPredictor()
        print("✓ Model loaded successfully on startup")
    except FileNotFoundError:
        print("⚠ Warning: Model not found. Please train the model first.")
        predictor = None
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Energy Demand ML API",
    description="API for energy demand prediction using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Setup templates and static files.
# The static dir is created if missing so a fresh clone / rebuilt container
# never crashes at import time (StaticFiles raises if the directory is absent,
# and git does not track empty directories).
STATIC_DIR = PROJECT_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Cache-busting token appended to every /static URL (?v=...) so browsers pick
# up new CSS/JS immediately after each deploy instead of serving stale cached
# assets. Render injects RENDER_GIT_COMMIT; locally we fall back to app start
# time (cache is at worst per-process-restart).
ASSET_VERSION = (os.environ.get("RENDER_GIT_COMMIT") or "")[:8] or str(int(time.time()))
templates.env.globals["asset_v"] = ASSET_VERSION

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

class WhatIfRequest(BaseModel):
    """Base input plus which dimension to sweep, for what-if charts"""
    demand_forecast_mw: float
    net_generation_mw: float
    total_interchange_mw: float
    hour_number: int = Field(..., ge=1, le=25)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    balancing_authority: str
    sub_region: str
    season: str
    sweep_by: str = Field(..., description="hour | day_of_week | month | balancing_authority")

# Human-readable labels for model feature names (used by /api/feature-importance)
FEATURE_LABELS = {
    'Demand Forecast (MW)': 'Demand Forecast',
    'Net Generation (MW)': 'Net Generation',
    'Total Interchange (MW)': 'Total Interchange',
    'Hour Number': 'Hour Number',
    'hour': 'Hour of Day',
    'day_of_week': 'Day of Week',
    'month': 'Month',
    'Balancing Authority_encoded': 'Balancing Authority',
    'Sub-Region_encoded': 'Sub-Region',
    'season_encoded': 'Season',
}

DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _to_model_input(data: Dict) -> Dict:
    """Map API field names to the column names EnergyDemandPredictor expects"""
    return {
        'Demand Forecast (MW)': data['demand_forecast_mw'],
        'Net Generation (MW)': data['net_generation_mw'],
        'Total Interchange (MW)': data['total_interchange_mw'],
        'Hour Number': data['hour_number'],
        'hour': data['hour'],
        'day_of_week': data['day_of_week'],
        'month': data['month'],
        'Balancing Authority': data['balancing_authority'],
        'Sub-Region': data['sub_region'],
        'season': data['season'],
    }

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Landing page"""
    return templates.TemplateResponse(request, "index.html", {
        "title": "Energy Demand ML API",
        "model_loaded": predictor is not None
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
        input_data = _to_model_input(request.model_dump())

        # Make prediction
        result = predictor.predict_with_confidence(input_data)

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Endpoint 2b: Feature importance (powers the model-insight chart on the dashboard)
@app.get("/api/feature-importance")
async def feature_importance():
    """Return the trained model's feature importances for charting"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using POST /train"
        )

    try:
        importances = predictor.model.feature_importances_
        names = predictor.feature_names

        features = [
            {"feature": FEATURE_LABELS.get(name, name), "importance": float(score)}
            for name, score in zip(names, importances)
        ]
        features.sort(key=lambda f: f["importance"], reverse=True)

        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")

# Endpoint 2c: What-if sweep (powers the interactive line/bar chart on the dashboard)
@app.post("/api/whatif")
async def whatif(request: WhatIfRequest):
    """
    Hold all inputs fixed except one dimension, sweep it across its valid
    range, and return a prediction for each step - used to draw an
    interactive what-if chart client-side.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using POST /train"
        )

    base = request.model_dump()
    sweep_by = base.pop("sweep_by")

    if sweep_by == "hour":
        sweep_values = [(h, str(h)) for h in range(24)]
        field = "hour"
    elif sweep_by == "day_of_week":
        sweep_values = [(d, DAY_LABELS[d]) for d in range(7)]
        field = "day_of_week"
    elif sweep_by == "month":
        sweep_values = [(m, MONTH_LABELS[m - 1]) for m in range(1, 13)]
        field = "month"
    elif sweep_by == "balancing_authority":
        try:
            classes = list(predictor.label_encoders["Balancing Authority"].classes_)
        except Exception:
            classes = [base["balancing_authority"]]
        sweep_values = [(c, c) for c in classes]
        field = "balancing_authority"
    else:
        raise HTTPException(status_code=400, detail="sweep_by must be one of: hour, day_of_week, month, balancing_authority")

    try:
        points = []
        for value, label in sweep_values:
            row = dict(base)
            row[field] = value
            if sweep_by == "hour":
                row["hour_number"] = value + 1  # keep hour_number consistent with hour
            input_data = _to_model_input(row)
            result = predictor.predict_with_confidence(input_data)
            points.append({
                "label": label,
                "predicted_demand_mw": result["predicted_demand_mw"],
                "lower_bound_mw": result["lower_bound_mw"],
                "upper_bound_mw": result["upper_bound_mw"],
            })

        return {"sweep_by": sweep_by, "points": points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-if error: {str(e)}")

# Endpoint 2d: Metadata for building dropdowns (known balancing authorities / seasons)
@app.get("/api/metadata")
async def metadata():
    """Known categorical values, used to populate dashboard dropdowns"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using POST /train"
        )

    try:
        return {
            "balancing_authorities": sorted(list(predictor.label_encoders["Balancing Authority"].classes_)),
            "sub_regions": sorted(list(predictor.label_encoders["Sub-Region"].classes_)),
            "seasons": sorted(list(predictor.label_encoders["season"].classes_)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata error: {str(e)}")

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

    # Known categorical values, rendered server-side into <select>/<datalist>
    # options. Falls back to a fixed list if the model isn't loaded yet, so
    # the form always renders correctly.
    if predictor is not None:
        try:
            balancing_authorities = sorted(list(predictor.label_encoders["Balancing Authority"].classes_))
            sub_regions = sorted(list(predictor.label_encoders["Sub-Region"].classes_))
            seasons = sorted(list(predictor.label_encoders["season"].classes_))
        except Exception:
            balancing_authorities, sub_regions, seasons = [], [], ["winter", "spring", "summer", "fall"]
    else:
        balancing_authorities, sub_regions, seasons = [], [], ["winter", "spring", "summer", "fall"]

    return templates.TemplateResponse(request, "dashboard.html", {
        "title": "Energy Demand Prediction Dashboard",
        "model_status": model_status,
        "example_data": example_data,
        "balancing_authorities": balancing_authorities,
        "sub_regions": sub_regions,
        "seasons": seasons
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
            "/api/feature-importance": "GET - Model feature importances",
            "/api/whatif": "POST - Sweep one input dimension for a what-if chart",
            "/api/metadata": "GET - Known categorical values for dashboard dropdowns",
            "/dashboard": "Interactive dashboard",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)"
        },
        "model_loaded": predictor is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
