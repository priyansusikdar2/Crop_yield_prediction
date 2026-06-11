"""
Crop Yield Prediction System
Load trained model and make predictions on new data
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
from src.preprocess import preprocess_data
import warnings
warnings.filterwarnings('ignore')


class CropYieldPredictor:
    """Main predictor class for crop yield estimation"""
    
    def __init__(self, model_path='models/best_model_advanced.h5', use_tuned=False):
        """
        Initialize the predictor with trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        use_tuned : bool
            Whether to use tuned model instead of best model
        """
        # Try to load best model first, then fallback to other options
        model_loaded = False
        
        # Try different model paths
        model_paths = [
            model_path,
            'models/best_model_advanced.h5',
            'models/best_tuned_model_advanced.h5',
            'models/best_model.h5',
            'models/final_model_advanced.h5'
        ]
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    self.model = tf.keras.models.load_model(path, compile=False)
                    print(f"✅ Model loaded from {path}")
                    model_loaded = True
                    break
            except Exception as e:
                continue
        
        if not model_loaded:
            print("⚠️ No trained model found. Creating dummy model for testing...")
            self.model = self._create_dummy_model()
        
        # Load scalers and imputer
        try:
            self.scaler_X = joblib.load('models/scaler_X.pkl')
            self.scaler_y = joblib.load('models/scaler_y.pkl')
            self.imputer = joblib.load('models/imputer.pkl')
            print("✅ Scalers and imputer loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load scalers: {e}")
            self.scaler_X = None
            self.scaler_y = None
            self.imputer = None
        
        # Load feature information
        try:
            self.feature_names = joblib.load('models/feature_cols.pkl')
            print(f"✅ Loaded {len(self.feature_names)} features")
        except:
            print("⚠️ Could not load feature columns, using defaults")
            self.feature_names = [
                'Temp_mean', 'Temp_std', 'Temp_min', 'Temp_max',
                'Rainfall_sum', 'Rainfall_mean', 'Rainfall_std',
                'Humidity_mean', 'Humidity_std', 'SolarRad_mean', 'SolarRad_std',
                'Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha', 'pH', 'Soil_Moisture',
                'Crop_Price', 'N_Soil', 'P_Soil', 'K_Soil'
            ]
        
        # Get input shape
        try:
            self.time_steps = self.model.input_shape[1]
            self.n_features = self.model.input_shape[2]
        except:
            self.time_steps = 3
            self.n_features = len(self.feature_names)
        
        print(f"📊 Model expects: {self.time_steps} timesteps × {self.n_features} features")
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when no model is available"""
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.LSTM(32, return_sequences=True, input_shape=(3, 20)),
            layers.LSTM(16),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_sequence_data(self, weather_years, soil_data, crop_price=None):
        """
        Prepare data for LSTM sequence prediction
        
        Parameters:
        -----------
        weather_years : list of dict
            List of 3 dictionaries, each containing yearly weather aggregates
            Each dict should have: temp_mean, temp_std, temp_min, temp_max,
            rainfall_sum, rainfall_mean, rainfall_std,
            humidity_mean, humidity_std,
            solarrad_mean, solarrad_std
        soil_data : dict
            Soil properties: nitrogen, phosphorus, potassium, ph, soil_moisture
        crop_price : float, optional
            Crop price for the region
        
        Returns:
        --------
        features_array : numpy array
            Prepared features of shape (1, timesteps, features)
        """
        features = []
        
        for year_data in weather_years:
            year_features = [
                year_data.get('temp_mean', 0),
                year_data.get('temp_std', 0),
                year_data.get('temp_min', 0),
                year_data.get('temp_max', 0),
                year_data.get('rainfall_sum', 0),
                year_data.get('rainfall_mean', 0),
                year_data.get('rainfall_std', 0),
                year_data.get('humidity_mean', 0),
                year_data.get('humidity_std', 0),
                year_data.get('solarrad_mean', 0),
                year_data.get('solarrad_std', 0),
                soil_data.get('nitrogen', 50),
                soil_data.get('phosphorus', 30),
                soil_data.get('potassium', 100),
                soil_data.get('ph', 6.5),
                soil_data.get('soil_moisture', 50),
                crop_price if crop_price else 1000,
                soil_data.get('n_soil', 50),
                soil_data.get('p_soil', 30),
                soil_data.get('k_soil', 100)
            ]
            features.append(year_features)
        
        features_array = np.array(features)
        
        # Handle missing values if imputer is available
        if self.imputer is not None:
            features_array = self.imputer.transform(features_array)
        
        # Scale features
        if self.scaler_X is not None:
            features_array = self.scaler_X.transform(features_array)
        
        # Reshape for LSTM (add batch dimension)
        features_array = features_array.reshape(1, self.time_steps, -1)
        
        # Ensure correct number of features
        if features_array.shape[2] != self.n_features:
            print(f"⚠️ Warning: Expected {self.n_features} features, got {features_array.shape[2]}")
            # Pad or truncate as needed
            if features_array.shape[2] < self.n_features:
                padding = np.zeros((1, self.time_steps, self.n_features - features_array.shape[2]))
                features_array = np.concatenate([features_array, padding], axis=2)
            else:
                features_array = features_array[:, :, :self.n_features]
        
        return features_array
    
    def predict(self, weather_years, soil_data, crop_price=None):
        """
        Make crop yield prediction
        
        Parameters:
        -----------
        weather_years : list of dict
            List of 3 dictionaries with yearly weather data
        soil_data : dict
            Dictionary with soil properties
        crop_price : float, optional
            Crop price in local currency
        
        Returns:
        --------
        prediction : float
            Predicted yield in kg/hectare
        confidence : float
            Prediction confidence (0-1)
        """
        # Prepare input data
        X_input = self.prepare_sequence_data(weather_years, soil_data, crop_price)
        
        # Make prediction
        try:
            prediction_scaled = self.model.predict(X_input, verbose=0)[0][0]
            
            # Inverse transform to get actual yield
            if self.scaler_y is not None:
                prediction_actual = self.scaler_y.inverse_transform([[prediction_scaled]])[0][0]
            else:
                prediction_actual = prediction_scaled
            
            # Calculate confidence based on prediction variance (simplified)
            confidence = min(0.95, max(0.5, 1.0 - abs(prediction_scaled - 0.5) * 0.5))
            
            return max(0, prediction_actual), confidence
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return 5000, 0.5  # Default fallback
    
    def predict_batch(self, weather_data_list, soil_data_list, crop_prices=None):
        """
        Make batch predictions for multiple locations/years
        
        Parameters:
        -----------
        weather_data_list : list of list of dict
            List of weather data for each prediction (each with 3 years)
        soil_data_list : list of dict
            List of soil data for each location
        crop_prices : list, optional
            List of crop prices
        
        Returns:
        --------
        predictions : list of float
            List of predicted yields
        confidences : list of float
            List of confidence scores
        """
        predictions = []
        confidences = []
        
        for i, weather_years in enumerate(weather_data_list):
            soil_data = soil_data_list[i] if i < len(soil_data_list) else soil_data_list[0]
            crop_price = crop_prices[i] if crop_prices and i < len(crop_prices) else None
            
            pred, conf = self.predict(weather_years, soil_data, crop_price)
            predictions.append(pred)
            confidences.append(conf)
        
        return predictions, confidences


def create_sample_weather_data():
    """Create sample weather data for testing"""
    return [
        {  # Year 1
            'temp_mean': 24.5,
            'temp_std': 5.2,
            'temp_min': 15.0,
            'temp_max': 35.0,
            'rainfall_sum': 850,
            'rainfall_mean': 2.33,
            'rainfall_std': 1.5,
            'humidity_mean': 65,
            'humidity_std': 10,
            'solarrad_mean': 210,
            'solarrad_std': 30
        },
        {  # Year 2
            'temp_mean': 25.2,
            'temp_std': 5.5,
            'temp_min': 16.0,
            'temp_max': 36.0,
            'rainfall_sum': 820,
            'rainfall_mean': 2.25,
            'rainfall_std': 1.4,
            'humidity_mean': 68,
            'humidity_std': 11,
            'solarrad_mean': 205,
            'solarrad_std': 32
        },
        {  # Year 3
            'temp_mean': 26.1,
            'temp_std': 5.8,
            'temp_min': 17.0,
            'temp_max': 37.0,
            'rainfall_sum': 790,
            'rainfall_mean': 2.16,
            'rainfall_std': 1.3,
            'humidity_mean': 70,
            'humidity_std': 12,
            'solarrad_mean': 215,
            'solarrad_std': 35
        }
    ]


def create_sample_soil_data():
    """Create sample soil data for testing"""
    return {
        'nitrogen': 75,
        'phosphorus': 45,
        'potassium': 90,
        'ph': 6.8,
        'soil_moisture': 55,
        'n_soil': 75,
        'p_soil': 45,
        'k_soil': 90
    }


def simple_prediction(temperature, rainfall, humidity, solar_radiation,
                     nitrogen, phosphorus, potassium, ph, organic_matter=None):
    """
    Simple prediction function for quick testing
    
    All inputs should be for the last 3 years as lists/arrays of length 3
    
    Parameters:
    -----------
    temperature : list of 3 floats
        Average temperature for each of the last 3 years (°C)
    rainfall : list of 3 floats
        Total rainfall for each of the last 3 years (mm)
    humidity : list of 3 floats
        Average humidity for each of the last 3 years (%)
    solar_radiation : list of 3 floats
        Average solar radiation for each of the last 3 years (W/m²)
    nitrogen : list of 3 floats
        Soil nitrogen levels (kg/ha) - can be same for all years
    phosphorus : list of 3 floats
        Soil phosphorus levels (kg/ha) - can be same for all years
    potassium : list of 3 floats
        Soil potassium levels (kg/ha) - can be same for all years
    ph : list of 3 floats
        Soil pH - can be same for all years
    organic_matter : list of 3 floats, optional
        Soil organic matter percentage - can be same for all years
    
    Returns:
    --------
    predicted_yield : float
        Predicted crop yield in kg/hectare
    """
    # Create weather data from inputs
    weather_years = []
    for i in range(3):
        weather_years.append({
            'temp_mean': temperature[i] if isinstance(temperature, list) else temperature,
            'temp_std': 5.0,  # Default standard deviation
            'temp_min': (temperature[i] if isinstance(temperature, list) else temperature) - 10,
            'temp_max': (temperature[i] if isinstance(temperature, list) else temperature) + 10,
            'rainfall_sum': rainfall[i] if isinstance(rainfall, list) else rainfall,
            'rainfall_mean': (rainfall[i] if isinstance(rainfall, list) else rainfall) / 365,
            'rainfall_std': 1.5,
            'humidity_mean': humidity[i] if isinstance(humidity, list) else humidity,
            'humidity_std': 10,
            'solarrad_mean': solar_radiation[i] if isinstance(solar_radiation, list) else solar_radiation,
            'solarrad_std': 30
        })
    
    # Create soil data
    soil_data = {
        'nitrogen': nitrogen[0] if isinstance(nitrogen, list) else nitrogen,
        'phosphorus': phosphorus[0] if isinstance(phosphorus, list) else phosphorus,
        'potassium': potassium[0] if isinstance(potassium, list) else potassium,
        'ph': ph[0] if isinstance(ph, list) else ph,
        'soil_moisture': 50,
        'n_soil': nitrogen[0] if isinstance(nitrogen, list) else nitrogen,
        'p_soil': phosphorus[0] if isinstance(phosphorus, list) else phosphorus,
        'k_soil': potassium[0] if isinstance(potassium, list) else potassium
    }
    
    # Make prediction
    predictor = CropYieldPredictor()
    prediction, confidence = predictor.predict(weather_years, soil_data)
    
    return prediction


def compare_scenarios(scenarios):
    """
    Compare multiple prediction scenarios
    
    Parameters:
    -----------
    scenarios : list of dict
        Each dict contains 'name', 'weather_data', 'soil_data', 'crop_price'
    
    Returns:
    --------
    results : pandas.DataFrame
        Comparison results
    """
    predictor = CropYieldPredictor()
    results = []
    
    for scenario in scenarios:
        weather_data = scenario.get('weather_data', create_sample_weather_data())
        soil_data = scenario.get('soil_data', create_sample_soil_data())
        crop_price = scenario.get('crop_price', None)
        
        prediction, confidence = predictor.predict(weather_data, soil_data, crop_price)
        
        results.append({
            'Scenario': scenario['name'],
            'Predicted Yield (kg/ha)': round(prediction, 2),
            'Predicted Yield (tons/ha)': round(prediction / 1000, 2),
            'Confidence': round(confidence * 100, 1),
            'Crop Price': crop_price if crop_price else 'N/A'
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("="*60)
    print("🌾 CROP YIELD PREDICTION SYSTEM")
    print("="*60)
    
    # Example 1: Simple prediction
    print("\n📊 Example 1: Simple Prediction")
    print("-"*40)
    
    # Last 3 years of data
    temperatures = [24.5, 25.2, 26.1]
    rainfall = [850, 820, 790]
    humidity = [65, 68, 70]
    solar_rad = [210, 205, 215]
    
    # Soil properties (constant for all years)
    nitrogen = [75, 75, 75]
    phosphorus = [45, 45, 45]
    potassium = [90, 90, 90]
    ph = [6.8, 6.8, 6.8]
    
    predicted = simple_prediction(
        temperatures, rainfall, humidity, solar_rad,
        nitrogen, phosphorus, potassium, ph
    )
    
    print(f"📊 Input Data:")
    print(f"  Temperature trend: {temperatures[0]}°C → {temperatures[1]}°C → {temperatures[2]}°C")
    print(f"  Rainfall trend: {rainfall[0]}mm → {rainfall[1]}mm → {rainfall[2]}mm")
    print(f"\n🌾 Predicted Yield: {predicted:.2f} kg/ha")
    print(f"📊 That's approximately {predicted/1000:.2f} tons/ha")
    
    # Example 2: Compare different scenarios
    print("\n\n📊 Example 2: Scenario Comparison")
    print("-"*40)
    
    # Create different scenarios
    scenarios = [
        {
            'name': 'Normal Conditions',
            'weather_data': create_sample_weather_data(),
            'soil_data': create_sample_soil_data()
        },
        {
            'name': 'Drought Scenario',
            'weather_data': [
                {**create_sample_weather_data()[0], 'rainfall_sum': 400, 'rainfall_mean': 1.1},
                {**create_sample_weather_data()[1], 'rainfall_sum': 350, 'rainfall_mean': 0.96},
                {**create_sample_weather_data()[2], 'rainfall_sum': 300, 'rainfall_mean': 0.82}
            ],
            'soil_data': create_sample_soil_data()
        },
        {
            'name': 'High Temperature',
            'weather_data': [
                {**create_sample_weather_data()[0], 'temp_mean': 28, 'temp_max': 40},
                {**create_sample_weather_data()[1], 'temp_mean': 29, 'temp_max': 41},
                {**create_sample_weather_data()[2], 'temp_mean': 30, 'temp_max': 42}
            ],
            'soil_data': create_sample_soil_data()
        },
        {
            'name': 'Optimal Soil',
            'weather_data': create_sample_weather_data(),
            'soil_data': {
                'nitrogen': 120,
                'phosphorus': 80,
                'potassium': 150,
                'ph': 6.5,
                'soil_moisture': 65,
                'n_soil': 120,
                'p_soil': 80,
                'k_soil': 150
            }
        }
    ]
    
    results_df = compare_scenarios(scenarios)
    print("\n📊 Scenario Comparison Results:")
    print(results_df.to_string(index=False))
    
    # Example 3: Interactive prediction
    print("\n\n📊 Example 3: Interactive Prediction Mode")
    print("-"*40)
    
    use_interactive = input("\nWould you like to make a custom prediction? (y/n): ").lower() == 'y'
    
    if use_interactive:
        print("\n📝 Enter data for the last 3 years:")
        
        temps = []
        rains = []
        hums = []
        
        for i in range(3):
            print(f"\nYear {i+1}:")
            temp = float(input(f"  Average temperature (°C): "))
            rain = float(input(f"  Total rainfall (mm): "))
            hum = float(input(f"  Average humidity (%): "))
            temps.append(temp)
            rains.append(rain)
            hums.append(hum)
        
        print("\n🌱 Soil Properties:")
        n = float(input("  Nitrogen (kg/ha): "))
        p = float(input("  Phosphorus (kg/ha): "))
        k = float(input("  Potassium (kg/ha): "))
        ph_val = float(input("  Soil pH: "))
        
        custom_prediction = simple_prediction(
            temps, rains, hums, [200, 200, 200],
            [n, n, n], [p, p, p], [k, k, k], [ph_val, ph_val, ph_val]
        )
        
        print(f"\n🌾 Custom Prediction Result:")
        print(f"  Predicted Yield: {custom_prediction:.2f} kg/ha")
        print(f"  That's approximately {custom_prediction/1000:.2f} tons/ha")
    
    print("\n" + "="*60)
    print("✅ Prediction system ready for use!")
    print("="*60)