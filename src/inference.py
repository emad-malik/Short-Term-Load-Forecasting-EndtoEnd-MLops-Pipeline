"""
Inference Module - Making Predictions
Logic for loading models and making predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Union, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = PROJECT_ROOT / "models"

class EnergyDemandPredictor:
    """Class for making energy demand predictions"""
    
    def __init__(self):
        """Initialize the predictor by loading the model and artifacts"""
        self.model = None
        self.feature_names = None
        self.label_encoders = None
        self.metrics = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and associated artifacts"""
        model_path = MODELS_PATH / "xgboost_demand_model.pkl"
        features_path = MODELS_PATH / "feature_names.pkl"
        encoders_path = MODELS_PATH / "label_encoders.pkl"
        metrics_path = MODELS_PATH / "model_metrics.pkl"
        
        # Check if model exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please train the model first by running train.py"
            )
        
        # Load all artifacts
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        self.label_encoders = joblib.load(encoders_path)
        self.metrics = joblib.load(metrics_path)
        
        print("✓ Model loaded successfully!")
        print(f"Model performance (Test set):")
        print(f"  - MAE: {self.metrics['test_mae']:.2f} MW")
        print(f"  - RMSE: {self.metrics['test_rmse']:.2f} MW")
        print(f"  - R²: {self.metrics['test_r2']:.4f}")
    
    def prepare_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Prepare input data for prediction
        
        Args:
            input_data: Dictionary containing feature values
            
        Returns:
            DataFrame ready for prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col + '_encoded'] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories by using the most frequent class
                    print(f"Warning: Unseen category in {col}, using default encoding")
                    df[col + '_encoded'] = 0
        
        # Select only the features used in training
        try:
            X = df[self.feature_names]
        except KeyError as e:
            missing_features = set(self.feature_names) - set(df.columns)
            raise ValueError(f"Missing required features: {missing_features}")
        
        return X
    
    def predict(self, input_data: Union[Dict, List[Dict]]) -> Union[float, List[float]]:
        """
        Make prediction(s) for energy demand
        
        Args:
            input_data: Dictionary or list of dictionaries containing feature values
            
        Returns:
            Predicted demand in MW (single value or list)
        """
        # Handle single input
        if isinstance(input_data, dict):
            X = self.prepare_input(input_data)
            prediction = self.model.predict(X)[0]
            return float(prediction)
        
        # Handle batch input
        elif isinstance(input_data, list):
            predictions = []
            for data in input_data:
                X = self.prepare_input(data)
                pred = self.model.predict(X)[0]
                predictions.append(float(pred))
            return predictions
        
        else:
            raise ValueError("input_data must be a dictionary or list of dictionaries")
    
    def predict_with_confidence(self, input_data: Dict) -> Dict:
        """
        Make prediction with additional information
        
        Args:
            input_data: Dictionary containing feature values
            
        Returns:
            Dictionary with prediction and metadata
        """
        prediction = self.predict(input_data)
        
        # Calculate approximate confidence interval based on test RMSE
        confidence_interval = 1.96 * self.metrics['test_rmse']  # 95% CI
        
        return {
            'predicted_demand_mw': prediction,
            'lower_bound_mw': prediction - confidence_interval,
            'upper_bound_mw': prediction + confidence_interval,
            'model_test_mae': self.metrics['test_mae'],
            'model_test_rmse': self.metrics['test_rmse'],
            'model_test_r2': self.metrics['test_r2']
        }

def predict_single(input_data: Dict) -> float:
    """
    Convenience function for single prediction
    
    Args:
        input_data: Dictionary containing feature values
        
    Returns:
        Predicted demand in MW
    """
    predictor = EnergyDemandPredictor()
    return predictor.predict(input_data)

def predict_batch(input_data: List[Dict]) -> List[float]:
    """
    Convenience function for batch prediction
    
    Args:
        input_data: List of dictionaries containing feature values
        
    Returns:
        List of predicted demands in MW
    """
    predictor = EnergyDemandPredictor()
    return predictor.predict(input_data)

# Example usage
if __name__ == "__main__":
    # Example input data
    example_input = {
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
    
    print("="*60)
    print("Energy Demand Prediction - Inference Example")
    print("="*60)
    print("\nInput data:")
    for key, value in example_input.items():
        print(f"  {key}: {value}")
    
    try:
        # Initialize predictor
        predictor = EnergyDemandPredictor()
        
        # Make prediction
        print("\n" + "-"*60)
        print("Making prediction...")
        result = predictor.predict_with_confidence(example_input)
        
        print("\nPrediction Results:")
        print(f"  Predicted Demand: {result['predicted_demand_mw']:.2f} MW")
        print(f"  95% Confidence Interval: [{result['lower_bound_mw']:.2f}, {result['upper_bound_mw']:.2f}] MW")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the training pipeline first:")
        print("  python src/train.py")
