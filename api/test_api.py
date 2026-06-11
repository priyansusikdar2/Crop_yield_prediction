"""
API Testing Script
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_predict():
    """Test prediction endpoint"""
    
    # Sample weather data for 3 years
    weather_data = [
        {
            "temp_mean": 24.5,
            "temp_std": 5.2,
            "temp_min": 15.0,
            "temp_max": 35.0,
            "rainfall_sum": 850,
            "rainfall_mean": 2.33,
            "rainfall_std": 1.5,
            "humidity_mean": 65,
            "humidity_std": 10,
            "solarrad_mean": 210,
            "solarrad_std": 30
        },
        {
            "temp_mean": 25.2,
            "temp_std": 5.5,
            "temp_min": 16.0,
            "temp_max": 36.0,
            "rainfall_sum": 820,
            "rainfall_mean": 2.25,
            "rainfall_std": 1.4,
            "humidity_mean": 68,
            "humidity_std": 11,
            "solarrad_mean": 205,
            "solarrad_std": 32
        },
        {
            "temp_mean": 26.1,
            "temp_std": 5.8,
            "temp_min": 17.0,
            "temp_max": 37.0,
            "rainfall_sum": 790,
            "rainfall_mean": 2.16,
            "rainfall_std": 1.3,
            "humidity_mean": 70,
            "humidity_std": 12,
            "solarrad_mean": 215,
            "solarrad_std": 35
        }
    ]
    
    soil_data = {
        "nitrogen": 75,
        "phosphorus": 45,
        "potassium": 90,
        "ph": 6.8,
        "soil_moisture": 55,
        "organic_matter": 2.5
    }
    
    request_data = {
        "weather_data": weather_data,
        "soil_data": soil_data,
        "crop_type": "Wheat",
        "location": "Punjab, India",
        "crop_price": 2000
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=request_data)
    print(f"Prediction: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_batch_predict():
    """Test batch prediction endpoint"""
    
    weather_data = [
        {
            "temp_mean": 24.5,
            "temp_std": 5.2,
            "temp_min": 15.0,
            "temp_max": 35.0,
            "rainfall_sum": 850,
            "rainfall_mean": 2.33,
            "rainfall_std": 1.5,
            "humidity_mean": 65,
            "humidity_std": 10,
            "solarrad_mean": 210,
            "solarrad_std": 30
        },
        {
            "temp_mean": 25.2,
            "temp_std": 5.5,
            "temp_min": 16.0,
            "temp_max": 36.0,
            "rainfall_sum": 820,
            "rainfall_mean": 2.25,
            "rainfall_std": 1.4,
            "humidity_mean": 68,
            "humidity_std": 11,
            "solarrad_mean": 205,
            "solarrad_std": 32
        },
        {
            "temp_mean": 26.1,
            "temp_std": 5.8,
            "temp_min": 17.0,
            "temp_max": 37.0,
            "rainfall_sum": 790,
            "rainfall_mean": 2.16,
            "rainfall_std": 1.3,
            "humidity_mean": 70,
            "humidity_std": 12,
            "solarrad_mean": 215,
            "solarrad_std": 35
        }
    ]
    
    soil_data = {
        "nitrogen": 75,
        "phosphorus": 45,
        "potassium": 90,
        "ph": 6.8,
        "soil_moisture": 55
    }
    
    predictions = []
    for crop in ["Wheat", "Rice", "Maize"]:
        predictions.append({
            "weather_data": weather_data,
            "soil_data": soil_data,
            "crop_type": crop,
            "location": f"Test Location - {crop}"
        })
    
    request_data = {"predictions": predictions}
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=request_data)
    print(f"Batch Prediction: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        for result in results:
            print(f"  {result['crop_type']}: {result['predicted_yield_kg_ha']:.2f} kg/ha")
    return response.status_code == 200


def test_get_crops():
    """Test get crops endpoint"""
    response = requests.get(f"{BASE_URL}/crops")
    print(f"Get Crops: {response.status_code}")
    if response.status_code == 200:
        print(f"Available crops: {len(response.json()['crops'])}")
    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Model Info: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        print(f"  Model loaded: {info['model_loaded']}")
        print(f"  Time steps: {info['time_steps']}")
        print(f"  Features: {info['n_features']}")
    return response.status_code == 200


def test_history():
    """Test history endpoint"""
    response = requests.get(f"{BASE_URL}/history?limit=5")
    print(f"History: {response.status_code}")
    if response.status_code == 200:
        print(f"History entries: {len(response.json())}")
    return response.status_code == 200


if __name__ == "__main__":
    print("="*60)
    print("Testing Crop Yield Prediction API")
    print("="*60)
    
    tests = [
        ("Health Check", test_health),
        ("Get Crops", test_get_crops),
        ("Model Info", test_model_info),
        ("Single Prediction", test_predict),
        ("Batch Prediction", test_batch_predict),
        ("History", test_history)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📊 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")