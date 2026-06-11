import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from sklearn.impute import SimpleImputer

def load_and_merge():
    """Load and merge all data sources with proper handling"""
    try:
        # Load data
        weather = pd.read_csv('data/weather_data.csv')
        soil = pd.read_csv('data/soil_data.csv')
        crop = pd.read_csv('data/crop_data.csv')
        
        print(f"✅ Weather data: {weather.shape}")
        print(f"✅ Soil data: {soil.shape}")
        print(f"✅ Crop data: {crop.shape}")
        
        # Display column names
        print(f"\n📋 Weather columns: {list(weather.columns)}")
        print(f"📋 Soil columns: {list(soil.columns)}")
        print(f"📋 Crop columns: {list(crop.columns)}")
        
        # Process Weather Data
        weather = weather.rename(columns={
            'Date_Time': 'Date',
            'Precipitation_mm': 'Rainfall_mm'
        })
        
        # Convert date and extract year
        weather['Date'] = pd.to_datetime(weather['Date'], errors='coerce')
        weather['Year'] = weather['Date'].dt.year
        weather = weather.dropna(subset=['Year'])
        weather['Year'] = weather['Year'].astype(int)
        
        # Add estimated Solar Radiation
        if 'SolarRadiation_Wm2' not in weather.columns:
            # Estimate based on temperature and humidity
            weather['SolarRadiation_Wm2'] = 150 + (weather['Temperature_C'] - 20) * 10
            weather['SolarRadiation_Wm2'] = weather['SolarRadiation_Wm2'].clip(50, 350)
        
        print(f"📅 Weather years: {weather['Year'].min()} - {weather['Year'].max()}")
        print(f"📍 Unique weather locations: {weather['Location'].nunique()}")
        
        # Aggregate weather yearly
        weather_yearly = weather.groupby(['Year', 'Location']).agg({
            'Temperature_C': ['mean', 'std', 'min', 'max'],
            'Rainfall_mm': ['sum', 'mean', 'std'],
            'Humidity_pct': ['mean', 'std'],
            'SolarRadiation_Wm2': ['mean', 'std']
        }).reset_index()
        
        weather_yearly.columns = ['Year', 'Location', 
                                  'Temp_mean', 'Temp_std', 'Temp_min', 'Temp_max',
                                  'Rainfall_sum', 'Rainfall_mean', 'Rainfall_std',
                                  'Humidity_mean', 'Humidity_std',
                                  'SolarRad_mean', 'SolarRad_std']
        
        # Process Crop Data
        crop = crop.rename(columns={
            'STATE': 'Location',
            'CROP': 'Crop',
            'CROP_PRICE': 'Crop_Price',
            'N_SOIL': 'N_Soil',
            'P_SOIL': 'P_Soil', 
            'K_SOIL': 'K_Soil',
            'TEMPERATURE': 'Crop_Temp',
            'HUMIDITY': 'Crop_Humidity',
            'RAINFALL': 'Crop_Rainfall',
            'ph': 'Crop_pH',
            'SOIL_TYPE': 'Soil_Type'
        })
        
        # Create synthetic yield based on crop price and soil nutrients
        crop['Yield_kg_per_ha'] = (
            crop['Crop_Price'] * 10 + 
            crop['N_Soil'] * 15 +       
            crop['P_Soil'] * 8 +        
            crop['K_Soil'] * 5          
        ).clip(1000, 12000)
        
        # Create realistic years for crop data (2010-2023)
        locations = crop['Location'].unique()
        np.random.seed(42)
        
        crop_years = []
        for loc in locations:
            loc_mask = crop['Location'] == loc
            n_records = loc_mask.sum()
            # Generate years between 2010-2023 for each location
            years = np.random.choice(range(2010, 2024), n_records, replace=True)
            crop_years.extend(sorted(years))
        
        crop['Year'] = crop_years
        print(f"📅 Crop years: {crop['Year'].min()} - {crop['Year'].max()}")
        print(f"📍 Unique crop locations: {crop['Location'].nunique()}")
        
        # Process Soil Data - Create soil profiles for each crop location
        soil = soil.rename(columns={
            'Nitrogen': 'Nitrogen_kg_ha',
            'Phosphorus': 'Phosphorus_kg_ha',
            'Potassium': 'Potassium_kg_ha',
            'Temperature': 'Soil_Temp',
            'Moisture': 'Soil_Moisture',
            'Label': 'Soil_Type_Label',
            'pH': 'pH'
        })
        
        # Create soil data for each crop location
        soil_records = []
        locations_list = crop['Location'].unique()
        
        for i, loc in enumerate(locations_list):
            # Assign a soil type based on location index
            soil_idx = i % len(soil)
            soil_row = soil.iloc[soil_idx].copy()
            soil_row['Location'] = loc
            soil_records.append(soil_row)
        
        soil_df = pd.DataFrame(soil_records)
        print(f"✅ Soil data processed for {len(soil_df)} locations")
        
        # Merge data - Use cross join between crop and weather for matching years
        # First, get all unique years from crop data
        crop_years_unique = crop['Year'].unique()
        
        # Filter weather data to only years present in crop data
        weather_filtered = weather_yearly[weather_yearly['Year'].isin(crop_years_unique)]
        
        if len(weather_filtered) == 0:
            print("⚠️ No matching years between crop and weather data!")
            print("Creating synthetic weather data for crop years...")
            
            # Create synthetic weather data for each location and year
            synthetic_weather = []
            for loc in locations_list:
                for year in crop_years_unique:
                    # Use average weather from original data
                    base_weather = weather_yearly[weather_yearly['Location'] == weather_yearly['Location'].iloc[0]]
                    if len(base_weather) > 0:
                        synth_row = base_weather.iloc[0].copy()
                        synth_row['Year'] = year
                        synth_row['Location'] = loc
                        # Add some random variation
                        for col in ['Temp_mean', 'Temp_std', 'Temp_min', 'Temp_max', 
                                   'Rainfall_sum', 'Rainfall_mean', 'Rainfall_std',
                                   'Humidity_mean', 'Humidity_std', 
                                   'SolarRad_mean', 'SolarRad_std']:
                            if col in synth_row:
                                synth_row[col] = synth_row[col] * (1 + np.random.normal(0, 0.1))
                        synthetic_weather.append(synth_row)
            
            weather_filtered = pd.DataFrame(synthetic_weather)
        
        # Merge crop with weather
        merged = pd.merge(crop, weather_filtered, on=['Year', 'Location'], how='left')
        
        # Merge with soil
        merged = pd.merge(merged, soil_df, on='Location', how='left')
        
        # Fill missing weather data with column means
        weather_cols = ['Temp_mean', 'Temp_std', 'Temp_min', 'Temp_max',
                       'Rainfall_sum', 'Rainfall_mean', 'Rainfall_std',
                       'Humidity_mean', 'Humidity_std', 'SolarRad_mean', 'SolarRad_std']
        
        for col in weather_cols:
            if col in merged.columns and merged[col].isna().any():
                merged[col] = merged[col].fillna(merged[col].mean())
        
        # Fill missing soil data with column means
        soil_cols = ['Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha', 'pH', 'Soil_Moisture']
        for col in soil_cols:
            if col in merged.columns and merged[col].isna().any():
                merged[col] = merged[col].fillna(merged[col].mean())
        
        # Drop any remaining rows with missing values
        initial_shape = merged.shape
        merged = merged.dropna()
        print(f"✅ Dropped {initial_shape[0] - merged.shape[0]} rows with missing values")
        
        print(f"\n✅ Data merged successfully. Shape: {merged.shape}")
        print(f"📊 Years range: {merged['Year'].min()} - {merged['Year'].max()}")
        print(f"📍 Unique locations: {merged['Location'].nunique()}")
        print(f"🌾 Unique crops: {merged['Crop'].nunique()}")
        
        return merged
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_simple_sequences(df, feature_cols, target_col, seq_length=3):
    """Creates sliding window sequences"""
    X, y = [], []
    
    # Sort by year and location
    df_sorted = df.sort_values(['Location', 'Crop', 'Year'])
    
    # Group by location and crop
    grouped = df_sorted.groupby(['Location', 'Crop'])
    
    sequences_created = 0
    for (loc, crop_type), group in grouped:
        if len(group) <= seq_length:
            continue
        
        features = group[feature_cols].values
        targets = group[target_col].values
        
        for i in range(len(features) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(targets[i+seq_length])
            sequences_created += 1
    
    if len(X) == 0:
        print("❌ No sequences created with grouping. Using all data...")
        # Fallback: use all data without grouping
        df_sorted = df.sort_values('Year')
        features = df_sorted[feature_cols].values
        targets = df_sorted[target_col].values
        
        for i in range(len(features) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(targets[i+seq_length])
    
    print(f"✅ Created {len(X)} sequences")
    return np.array(X), np.array(y)

def preprocess_data():
    """Main preprocessing function"""
    
    print("="*60)
    print("🌾 CROP YIELD PREDICTION - DATA PREPROCESSING")
    print("="*60)
    
    # Load and merge data
    df = load_and_merge()
    
    if len(df) == 0:
        print("❌ No data after merging! Creating synthetic data for testing...")
        # Create synthetic data for testing
        n_samples = 1000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'Year': np.random.choice(range(2010, 2024), n_samples),
            'Location': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'Crop': np.random.choice(['Wheat', 'Rice', 'Maize', 'Soybean'], n_samples),
            'Temp_mean': np.random.normal(25, 5, n_samples),
            'Temp_std': np.random.normal(5, 2, n_samples),
            'Temp_min': np.random.normal(15, 5, n_samples),
            'Temp_max': np.random.normal(35, 5, n_samples),
            'Rainfall_sum': np.random.normal(800, 200, n_samples),
            'Rainfall_mean': np.random.normal(2, 1, n_samples),
            'Rainfall_std': np.random.normal(1, 0.5, n_samples),
            'Humidity_mean': np.random.normal(60, 15, n_samples),
            'Humidity_std': np.random.normal(10, 3, n_samples),
            'SolarRad_mean': np.random.normal(200, 50, n_samples),
            'SolarRad_std': np.random.normal(30, 10, n_samples),
            'Nitrogen_kg_ha': np.random.normal(100, 50, n_samples),
            'Phosphorus_kg_ha': np.random.normal(50, 25, n_samples),
            'Potassium_kg_ha': np.random.normal(200, 100, n_samples),
            'pH': np.random.normal(6.5, 0.5, n_samples),
            'Soil_Moisture': np.random.normal(50, 20, n_samples),
            'Crop_Price': np.random.normal(1000, 300, n_samples),
            'N_Soil': np.random.normal(100, 50, n_samples),
            'P_Soil': np.random.normal(50, 25, n_samples),
            'K_Soil': np.random.normal(200, 100, n_samples),
            'Yield_kg_per_ha': np.random.normal(5000, 1500, n_samples)
        })
        
        print(f"✅ Created synthetic data with {len(df)} samples")
    
    # Define available features
    available_features = []
    
    # Weather features
    weather_features = ['Temp_mean', 'Temp_std', 'Temp_min', 'Temp_max',
                       'Rainfall_sum', 'Rainfall_mean', 'Rainfall_std',
                       'Humidity_mean', 'Humidity_std',
                       'SolarRad_mean', 'SolarRad_std']
    
    # Soil features
    soil_features = ['Nitrogen_kg_ha', 'Phosphorus_kg_ha', 
                    'Potassium_kg_ha', 'pH', 'Soil_Moisture']
    
    # Crop features
    crop_features = ['Crop_Price', 'N_Soil', 'P_Soil', 'K_Soil']
    
    # Use only features that exist in the dataframe
    for feat in weather_features + soil_features + crop_features:
        if feat in df.columns:
            available_features.append(feat)
    
    feature_cols = available_features
    target_col = 'Yield_kg_per_ha'
    
    print(f"\n📊 Feature columns ({len(feature_cols)} features):")
    for i, f in enumerate(feature_cols):
        print(f"  {i+1}. {f}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    # Normalize features
    scaler_X = MinMaxScaler()
    df[feature_cols] = scaler_X.fit_transform(df[feature_cols])
    
    # Normalize target
    scaler_y = MinMaxScaler()
    df[target_col] = scaler_y.fit_transform(df[[target_col]])
    
    # Create sequences
    X, y = create_simple_sequences(df, feature_cols, target_col, seq_length=3)
    
    print(f"\n📊 Sequences created - X shape: {X.shape}, y shape: {y.shape}")
    
    if len(X) == 0:
        print("❌ No sequences created! Using non-sequential data...")
        # Fallback: use non-sequential data
        X = df[feature_cols].values
        y = df[target_col].values
        # Reshape to add sequence dimension (1)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        print(f"✅ Using non-sequential data - X shape: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"\n✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Save scalers and imputer
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print("✅ Scalers and imputer saved to models/")
    
    # Save data info
    data_info = {
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'input_shape': X_train.shape[1:]
    }
    joblib.dump(data_info, 'models/data_info.pkl')
    print("✅ Data info saved to models/data_info.pkl")
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_cols

if __name__ == "__main__":
    result = preprocess_data()
    if result[0] is not None and len(result[0]) > 0:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, features = result
        print(f"\n📊 Final data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        print(f"\n✅ Preprocessing completed successfully!")
        
        # Print sample of the data
        print(f"\n📊 Sample of preprocessed data:")
        print(f"  Input shape (first sample): {X_train[0].shape}")
        print(f"  Output range: {y_train.min():.3f} - {y_train.max():.3f}")
    else:
        print("\n❌ Preprocessing failed!")