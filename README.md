# Energy Demand ML - Deployment Pipeline

A production-ready machine learning pipeline for predicting energy demand using XGBoost.

## Project Structure

```
energy-ml-data/
├── data/
│   ├── raw/                # Raw CSV and JSON files
│   └── processed/          # Cleaned data (merged_energy_data.csv, weather_merged.csv)
├── models/                 # Trained models and artifacts
│   ├── xgboost_demand_model.pkl
│   ├── feature_names.pkl
│   ├── label_encoders.pkl
│   └── model_metrics.pkl
├── src/
│   ├── etl.py              # Data extraction & cleaning
│   ├── train.py            # Model training & saving
│   └── inference.py        # Prediction logic
├── app/
│   ├── main.py             # FastAPI application
│   ├── templates/          # HTML templates
│   └── static/             # Generated plots/images
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
└── Dockerfile              # Container configuration
```

## Quick Start

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run ETL Pipeline

Process raw data and create clean datasets:

```powershell
python src/etl.py
```

**Output:**
- `data/processed/merged_energy_data.csv` - Cleaned energy data (1.3M rows)
- `data/processed/weather_merged.csv` - Cleaned weather data (165K rows)

### 3. Train Model

Train the XGBoost model for demand prediction:

```powershell
python src/train.py
```

**Output:**
- `models/xgboost_demand_model.pkl` - Trained XGBoost model
- `models/feature_names.pkl` - List of features used
- `models/label_encoders.pkl` - Encoders for categorical features
- `models/model_metrics.pkl` - Model performance metrics

**Expected Performance:**
- Test MAE: ~XX MW
- Test RMSE: ~XX MW
- Test R²: ~0.XX

### 4. Make Predictions

Use the trained model for inference:

```python
from src.inference import EnergyDemandPredictor

# Initialize predictor
predictor = EnergyDemandPredictor()

# Prepare input data
input_data = {
    'Demand Forecast (MW)': 5000.0,
    'Net Generation (MW)': 4800.0,
    'Total Interchange (MW)': 200.0,
    'Hour Number': 14,
    'hour': 14,
    'day_of_week': 2,  # Wednesday
    'month': 7,  # July
    'Balancing Authority': 'CISO',
    'Sub-Region': 'PGAE',
    'season': 'summer'
}

# Get prediction with confidence interval
result = predictor.predict_with_confidence(input_data)
print(f"Predicted Demand: {result['predicted_demand_mw']:.2f} MW")
```

Or run the example:

```powershell
python src/inference.py
```

## Model Features

The XGBoost model uses the following features:

**Numeric Features:**
- `Demand Forecast (MW)` - Forecasted demand
- `Net Generation (MW)` - Net power generation
- `Total Interchange (MW)` - Power interchange
- `Hour Number` - Hour of the day
- `hour` - Hour (0-23)
- `day_of_week` - Day of week (0-6)
- `month` - Month (1-12)

**Categorical Features:**
- `Balancing Authority` - Power authority
- `Sub-Region` - Geographic sub-region
- `season` - Season (winter/spring/summer/fall)

## API Deployment

Run the FastAPI application:

```powershell
# Development mode
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access the API at: `http://localhost:8000`

### Available Endpoints

- **GET /** - Home page
- **GET /dashboard** - Interactive prediction dashboard
- **GET /health** - Health check and model metrics
- **POST /train** - Trigger model retraining (background task)
- **POST /predict/xgboost** - Make energy demand prediction
- **GET /docs** - Swagger UI documentation
- **GET /redoc** - ReDoc documentation

### Example API Request

```powershell
$body = @{
    demand_forecast_mw = 5000.0
    net_generation_mw = 4800.0
    total_interchange_mw = 200.0
    hour_number = 14
    hour = 14
    day_of_week = 2
    month = 7
    balancing_authority = "CISO"
    sub_region = "PGAE"
    season = "summer"
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/predict/xgboost `
    -Method POST `
    -ContentType "application/json" `
    -Body $body `
    -UseBasicParsing
```

## Docker Deployment

### Quick Start with Docker Compose

```powershell
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Build

```powershell
# Build the image
docker build -t energy-ml .

# Run the container
docker run -d -p 80:80 --name energy-ml-api energy-ml

# Test the API
Invoke-WebRequest -Uri http://localhost/health -UseBasicParsing
```

Access the containerized API at: `http://localhost`

## Data Pipeline

### ETL Process (`src/etl.py`)

1. **Extract**: Load raw EIA930 energy data and weather JSON files
2. **Transform**:
   - Drop columns with >10% missing values (subregion) or >20% (balance)
   - Merge subregion and balance data
   - Clean numeric columns (remove commas, convert types)
   - Fill missing values with grouped means
   - Engineer time features (hour, day_of_week, month, season)
3. **Load**: Save processed data to CSV files

### Training Process (`src/train.py`)

1. Load processed energy data
2. Prepare features (encode categoricals, select features)
3. Split data (80% train, 20% test)
4. Train XGBoost model with optimized hyperparameters
5. Evaluate performance (MAE, RMSE, R²)
6. Save model and artifacts

### Inference Process (`src/inference.py`)

1. Load trained model and artifacts
2. Prepare input data (encode categoricals, align features)
3. Make predictions
4. Return results with confidence intervals

## Development

### Adding New Features

1. Update `prepare_features()` in `src/train.py`
2. Retrain the model: `python src/train.py`
3. Test inference: `python src/inference.py`

### Model Tuning

Edit hyperparameters in `train_xgboost_model()` in `src/train.py`:

```python
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

## Requirements

- Python 3.11+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- fastapi >= 0.104.0
- uvicorn >= 0.24.0

## License

MIT License

## Contributors

- Your Team
