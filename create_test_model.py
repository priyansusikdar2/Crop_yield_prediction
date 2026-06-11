import tensorflow as tf
import numpy as np
import joblib
import os

os.makedirs('models', exist_ok=True)

print("Creating test model...")

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3, 20)),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.save('models/best_model_advanced.h5')
print("✅ Model saved to models/best_model_advanced.h5")

# Create dummy scalers
scaler_X = joblib.load('models/scaler_X.pkl') if os.path.exists('models/scaler_X.pkl') else None
scaler_y = joblib.load('models/scaler_y.pkl') if os.path.exists('models/scaler_y.pkl') else None

print("✅ Test model created successfully!")