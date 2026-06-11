"""
API Schemas for Crop Yield Prediction
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class WeatherData(BaseModel):
    """Weather data for a single year"""
    temp_mean: float = Field(..., description="Mean temperature (°C)", ge=-10, le=50)
    temp_std: float = Field(..., description="Standard deviation of temperature", ge=0, le=20)
    temp_min: float = Field(..., description="Minimum temperature (°C)", ge=-20, le=45)
    temp_max: float = Field(..., description="Maximum temperature (°C)", ge=0, le=55)
    rainfall_sum: float = Field(..., description="Total rainfall (mm)", ge=0, le=5000)
    rainfall_mean: float = Field(..., description="Mean daily rainfall (mm)", ge=0, le=50)
    rainfall_std: float = Field(..., description="Standard deviation of rainfall", ge=0, le=20)
    humidity_mean: float = Field(..., description="Mean humidity (%)", ge=0, le=100)
    humidity_std: float = Field(..., description="Standard deviation of humidity", ge=0, le=30)
    solarrad_mean: float = Field(..., description="Mean solar radiation (W/m²)", ge=0, le=500)
    solarrad_std: float = Field(..., description="Standard deviation of solar radiation", ge=0, le=100)
    
    @validator('temp_min', 'temp_max')
    def validate_temperature_range(cls, v, values):
        if 'temp_mean' in values and abs(v - values['temp_mean']) > 30:
            raise ValueError(f'Temperature too far from mean')
        return v


class SoilData(BaseModel):
    """Soil properties data"""
    nitrogen: float = Field(..., description="Nitrogen content (kg/ha)", ge=0, le=500)
    phosphorus: float = Field(..., description="Phosphorus content (kg/ha)", ge=0, le=300)
    potassium: float = Field(..., description="Potassium content (kg/ha)", ge=0, le=600)
    ph: float = Field(..., description="Soil pH", ge=3.5, le=9.5)
    soil_moisture: float = Field(50, description="Soil moisture (%)", ge=0, le=100)
    organic_matter: Optional[float] = Field(2.5, description="Organic matter (%)", ge=0, le=20)
    
    @validator('ph')
    def validate_ph(cls, v):
        if v < 4.0 or v > 9.0:
            raise ValueError(f'pH value {v} is outside optimal range (4.0-9.0)')
        return v


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    weather_data: List[WeatherData] = Field(..., min_items=3, max_items=3, description="3 years of weather data")
    soil_data: SoilData = Field(..., description="Soil properties")
    crop_type: Optional[str] = Field("Wheat", description="Type of crop")
    location: Optional[str] = Field(None, description="Location name")
    crop_price: Optional[float] = Field(None, description="Crop price per kg", ge=0)
    
    @validator('weather_data')
    def validate_weather_years(cls, v):
        if len(v) != 3:
            raise ValueError('Exactly 3 years of weather data required')
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    predictions: List[PredictionRequest] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    predicted_yield_kg_ha: float = Field(..., description="Predicted yield in kg/hectare")
    predicted_yield_tons_ha: float = Field(..., description="Predicted yield in tons/hectare")
    confidence_score: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    crop_type: str
    location: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_yield_kg_ha": 5234.56,
                "predicted_yield_tons_ha": 5.23,
                "confidence_score": 0.87,
                "crop_type": "Wheat",
                "location": "Punjab, India",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionHistory(BaseModel):
    """Historical prediction record"""
    id: str
    prediction: PredictionResponse
    input_summary: Dict[str, Any]
    created_at: datetime


# Crop types with their adjustment factors
class CropType(str, Enum):
    WHEAT = "Wheat"
    RICE = "Rice"
    MAIZE = "Maize"
    SOYBEAN = "Soybean"
    BARLEY = "Barley"
    SORGHUM = "Sorghum"
    MILLET = "Millet"
    COTTON = "Cotton"
    SUGARCANE = "Sugarcane"
    POTATO = "Potato"


CROP_FACTORS = {
    CropType.WHEAT: 1.0,
    CropType.RICE: 1.15,
    CropType.MAIZE: 0.95,
    CropType.SOYBEAN: 0.9,
    CropType.BARLEY: 0.85,
    CropType.SORGHUM: 0.8,
    CropType.MILLET: 0.75,
    CropType.COTTON: 0.7,
    CropType.SUGARCANE: 1.3,
    CropType.POTATO: 1.1,
}