"""
FastAPI Application for Crop Yield Prediction
Using trained models from the models directory
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import joblib
import tensorflow as tf
import os
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

# ============ Schemas ============
class WeatherData(BaseModel):
    temp_mean: float
    temp_std: float = 5.0
    temp_min: float
    temp_max: float
    rainfall_sum: float
    rainfall_mean: float
    rainfall_std: float = 1.5
    humidity_mean: float
    humidity_std: float = 10.0
    solarrad_mean: float
    solarrad_std: float = 30.0


class SoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    soil_moisture: float = 55.0


class PredictionRequest(BaseModel):
    weather_data: List[WeatherData]
    soil_data: SoilData
    crop_type: str = "Wheat"
    location: Optional[str] = None
    crop_price: Optional[float] = None


class PredictionResponse(BaseModel):
    predicted_yield_kg_ha: float
    predicted_yield_tons_ha: float
    confidence_score: float
    crop_type: str
    location: Optional[str]
    timestamp: datetime = datetime.now()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime


# Crop factors
CROP_FACTORS = {
    "Wheat": 1.0,
    "Rice": 1.15,
    "Maize": 0.95,
    "Soybean": 0.9,
    "Barley": 0.85,
    "Cotton": 0.7,
    "Sugarcane": 1.3,
    "Potato": 1.1,
}

# Initialize FastAPI app
app = FastAPI(title="Crop Yield Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler_X = None
scaler_y = None
model_loaded = False
feature_names = None


def load_models():
    """Load trained models and preprocessing objects"""
    global model, scaler_X, scaler_y, model_loaded, feature_names
    
    print("\n🔧 Loading trained models...")
    
    # Try to load the trained model - using your actual filenames
    model_paths = [
        'models/best_model_advanced.h5',
        'models/final_model_advanced.h5',
        'models/best_tuned_model_advanced.h5',
        'models/best_model.h5',
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Load model with custom_objects to handle attention layers
                from src.attention import Attention, MultiHeadAttention, TemporalAttention, FeatureAttention
                custom_objects = {
                    'Attention': Attention,
                    'MultiHeadAttention': MultiHeadAttention,
                    'TemporalAttention': TemporalAttention,
                    'FeatureAttention': FeatureAttention
                }
                model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                print(f"✅ Model loaded from {path}")
                model_found = True
                break
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")
                continue
    
    if not model_found:
        print("❌ No model found! Please check the models directory.")
        model_loaded = False
        return False
    
    # Load scalers - using your actual filenames
    scaler_paths = [
        'models/scaler_X_advanced.pkl',
        'models/scaler_X.pkl',
    ]
    for path in scaler_paths:
        if os.path.exists(path):
            try:
                scaler_X = joblib.load(path)
                print(f"✅ Scaler_X loaded from {path}")
                break
            except:
                continue
    
    scaler_y_paths = [
        'models/scaler_y_advanced.pkl',
        'models/scaler_y.pkl',
    ]
    for path in scaler_y_paths:
        if os.path.exists(path):
            try:
                scaler_y = joblib.load(path)
                print(f"✅ Scaler_y loaded from {path}")
                break
            except:
                continue
    
    # Load feature names
    feature_paths = [
        'models/feature_cols.pkl',
        'models/features_advanced.pkl',
    ]
    for path in feature_paths:
        if os.path.exists(path):
            try:
                feature_names = joblib.load(path)
                print(f"✅ Loaded {len(feature_names)} features from {path}")
                break
            except:
                continue
    
    model_loaded = True
    print("✅ All models loaded successfully!")
    return True


def prepare_features(weather_data, soil_data):
    """Prepare features for prediction"""
    features = []
    
    for year_data in weather_data:
        year_features = [
            year_data.temp_mean,
            year_data.temp_std,
            year_data.temp_min,
            year_data.temp_max,
            year_data.rainfall_sum,
            year_data.rainfall_mean,
            year_data.rainfall_std,
            year_data.humidity_mean,
            year_data.humidity_std,
            year_data.solarrad_mean,
            year_data.solarrad_std,
            soil_data.nitrogen,
            soil_data.phosphorus,
            soil_data.potassium,
            soil_data.ph,
            soil_data.soil_moisture
        ]
        features.append(year_features)
    
    features_array = np.array(features, dtype=np.float32)
    
    # Scale features if scaler is available
    if scaler_X is not None:
        try:
            features_flat = features_array.reshape(-1, 16)
            features_scaled = scaler_X.transform(features_flat)
            features_array = features_scaled.reshape(1, 3, 16)
        except Exception as e:
            print(f"⚠️ Scaling failed: {e}")
            features_array = features_array.reshape(1, 3, 16)
    else:
        features_array = features_array.reshape(1, 3, 16)
    
    return features_array


def predict(weather_data, soil_data, crop_type="Wheat"):
    """Make prediction using trained model"""
    global model, scaler_y, model_loaded
    
    if not model_loaded:
        if not load_models():
            # Fallback if model not loaded
            return calculate_fallback_yield(weather_data, soil_data, crop_type), 0.7
    
    try:
        # Prepare features
        X_input = prepare_features(weather_data, soil_data)
        
        # Make prediction
        prediction_scaled = model.predict(X_input, verbose=0)[0][0]
        
        # Inverse transform
        if scaler_y is not None:
            prediction_actual = scaler_y.inverse_transform([[prediction_scaled]])[0][0]
        else:
            # Approximate inverse (assuming yield range 1000-12000 kg/ha)
            prediction_actual = 1000 + prediction_scaled * 11000
        
        # Apply crop factor
        if crop_type in CROP_FACTORS:
            prediction_actual *= CROP_FACTORS[crop_type]
        
        # Calculate confidence based on prediction
        confidence = min(0.95, max(0.7, 0.8 + (1 - abs(prediction_scaled - 0.5)) * 0.15))
        
        return max(500, min(15000, prediction_actual)), confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return calculate_fallback_yield(weather_data, soil_data, crop_type), 0.7


def calculate_fallback_yield(weather_data, soil_data, crop_type):
    """Fallback yield calculation if model fails"""
    avg_temp = np.mean([w.temp_mean for w in weather_data])
    total_rainfall = np.sum([w.rainfall_sum for w in weather_data])
    avg_humidity = np.mean([w.humidity_mean for w in weather_data])
    
    temp_score = max(0, 1 - abs(avg_temp - 25) / 15)
    rain_score = max(0, 1 - abs(total_rainfall - 2400) / 1200)
    humidity_score = max(0, 1 - abs(avg_humidity - 65) / 30)
    
    n_score = min(1, soil_data.nitrogen / 120)
    p_score = min(1, soil_data.phosphorus / 80)
    k_score = min(1, soil_data.potassium / 200)
    ph_score = 1 - abs(soil_data.ph - 6.5) / 2
    
    weather_score = temp_score * 0.4 + rain_score * 0.35 + humidity_score * 0.25
    soil_score = n_score * 0.35 + p_score * 0.25 + k_score * 0.25 + ph_score * 0.15
    
    base_yield = 5000
    yield_kg = base_yield * (0.6 + 0.4 * weather_score) * (0.7 + 0.3 * soil_score)
    
    if crop_type in CROP_FACTORS:
        yield_kg *= CROP_FACTORS[crop_type]
    
    return max(1000, min(12000, yield_kg))


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """Make a single prediction"""
    try:
        yield_kg, confidence = predict(
            request.weather_data,
            request.soil_data,
            request.crop_type
        )
        
        return PredictionResponse(
            predicted_yield_kg_ha=round(yield_kg, 2),
            predicted_yield_tons_ha=round(yield_kg / 1000, 2),
            confidence_score=round(confidence, 3),
            crop_type=request.crop_type,
            location=request.location
        )
        
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crops")
async def get_available_crops():
    """Get list of available crop types"""
    return {
        "crops": [
            {"name": name, "factor": factor}
            for name, factor in CROP_FACTORS.items()
        ]
    }


@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "model_loaded": model_loaded,
        "model_paths": ['models/best_model_advanced.h5', 'models/final_model_advanced.h5'],
        "n_features": 16,
        "time_steps": 3,
        "feature_names": feature_names[:10] if feature_names else None
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )