"""
Training Module - Model Training and Saving
Handles model training and persistence for energy demand prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"

def load_processed_data():
    """Load the processed energy data"""
    print("Loading processed data...")
    energy_path = PROCESSED_DATA_PATH / "merged_energy_data.csv"
    
    if not energy_path.exists():
        raise FileNotFoundError(f"Processed data not found at {energy_path}. Please run ETL first.")
    
    df = pd.read_csv(energy_path)
    print(f"Loaded data shape: {df.shape}")
    return df

def prepare_features(df):
    """Prepare features for training"""
    print("Preparing features...")
    
    data = df.copy()
    
    # Select features for training
    # Numeric features
    numeric_features = [
        'Demand Forecast (MW)',
        'Net Generation (MW)',
        'Total Interchange (MW)',
        'Hour Number',
        'hour',
        'day_of_week',
        'month'
    ]
    
    # Categorical features
    categorical_features = [
        'Balancing Authority',
        'Sub-Region',
        'season'
    ]
    
    # Target variable
    target = 'Demand (MW) (Adjusted)'
    
    # Check if target exists
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    
    # Filter for available features
    available_numeric = [f for f in numeric_features if f in data.columns]
    available_categorical = [f for f in categorical_features if f in data.columns]
    
    print(f"Using {len(available_numeric)} numeric features: {available_numeric}")
    print(f"Using {len(available_categorical)} categorical features: {available_categorical}")
    
    # Encode categorical features
    label_encoders = {}
    for col in available_categorical:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # Combine all features
    encoded_categorical = [col + '_encoded' for col in available_categorical]
    all_features = available_numeric + encoded_categorical
    
    # Prepare X and y
    X = data[all_features]
    y = data[target]
    
    # Remove any rows with NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y, all_features, label_encoders

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model for demand prediction"""
    print("\nTraining XGBoost model...")
    
    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n=== Model Performance ===")
    print(f"Train MAE: {train_mae:.2f} MW")
    print(f"Train RMSE: {train_rmse:.2f} MW")
    print(f"Train R²: {train_r2:.4f}")
    print(f"\nTest MAE: {test_mae:.2f} MW")
    print(f"Test RMSE: {test_rmse:.2f} MW")
    print(f"Test R²: {test_r2:.4f}")
    
    return model, {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }

def save_model(model, feature_names, label_encoders, metrics):
    """Save the trained model and associated artifacts"""
    print("\nSaving model and artifacts...")
    
    # Ensure models directory exists
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model_path = MODELS_PATH / "xgboost_demand_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature names
    features_path = MODELS_PATH / "feature_names.pkl"
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved to: {features_path}")
    
    # Save label encoders
    encoders_path = MODELS_PATH / "label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved to: {encoders_path}")
    
    # Save metrics
    metrics_path = MODELS_PATH / "model_metrics.pkl"
    joblib.dump(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n✓ All artifacts saved successfully!")

def train_pipeline():
    """Main training pipeline"""
    print("="*60)
    print("Energy Demand Prediction - Training Pipeline")
    print("="*60)
    
    # Load data
    df = load_processed_data()
    
    # Prepare features
    X, y, feature_names, label_encoders = prepare_features(df)
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model, metrics = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # Save model and artifacts
    save_model(model, feature_names, label_encoders, metrics)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    train_pipeline()
