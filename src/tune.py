"""
Hyperparameter Tuning for Crop Yield Prediction Models
Using Keras Tuner for optimal model configuration
"""

import sys
import subprocess

# Check if keras_tuner is installed
try:
    import keras_tuner as kt
except ImportError:
    print("="*60)
    print("❌ Keras Tuner is not installed!")
    print("="*60)
    print("\nPlease install it using one of the following commands:")
    print("  pip install keras-tuner")
    print("\nAfter installation, run this script again.")
    print("="*60)
    sys.exit(1)

import tensorflow as tf
import numpy as np
import json
import os
from src.preprocess import preprocess_data
from src.model import build_advanced_model, build_bidirectional_model, build_weather_only_model
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_tunable_advanced_model(hp):
    """Model builder for advanced model with hyperparameter tuning"""
    
    # Hyperparameters to tune
    num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=3, default=2)
    
    lstm_units = []
    dropout_rates = []
    
    for i in range(num_lstm_layers):
        if i == 0:
            units = hp.Int(f'lstm_units_{i}', min_value=64, max_value=256, step=32, default=128)
        elif i == 1:
            units = hp.Int(f'lstm_units_{i}', min_value=32, max_value=128, step=16, default=64)
        else:
            units = hp.Int(f'lstm_units_{i}', min_value=16, max_value=64, step=8, default=32)
        
        dropout = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.3)
        
        lstm_units.append(units)
        dropout_rates.append(dropout)
    
    attention_type = hp.Choice('attention_type', values=['none', 'single', 'multihead', 'temporal', 'feature'], default='multihead')
    use_batch_norm = hp.Boolean('use_batch_norm', default=True)
    learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.0005, 0.001, 0.002, 0.005], default=0.001)
    # Fixed: Use float values only (0.0 instead of 0)
    l2_reg = hp.Choice('l2_reg', values=[0.0, 0.0001, 0.0005, 0.001, 0.005], default=0.001)
    
    # For multi-head attention, tune number of heads
    if attention_type == 'multihead':
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2, default=4)
    else:
        num_heads = 4
    
    # Get input shape from data
    try:
        result = preprocess_data()
        if result is not None and len(result) == 7:
            X_train, _, _, _, _, _, _ = result
            if X_train is not None and len(X_train) > 0:
                input_shape = (X_train.shape[1], X_train.shape[2])
            else:
                print("⚠️ No training data, using default shape")
                input_shape = (12, 50)
        else:
            print("⚠️ Preprocessing returned None, using default shape")
            input_shape = (12, 50)
    except Exception as e:
        print(f"⚠️ Could not get data, using default shape: {e}")
        input_shape = (12, 50)  # Default: 12 timesteps, 50 features
    
    model = build_advanced_model(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rates=dropout_rates,
        attention_type=attention_type,
        use_batch_norm=use_batch_norm,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
        num_heads=num_heads
    )
    
    return model


def build_tunable_bidirectional_model(hp):
    """Model builder for bidirectional LSTM model"""
    
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=16, default=64)
    attention_type = hp.Choice('attention_type', values=['none', 'single', 'multihead'], default='single')
    learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.0005, 0.001, 0.002], default=0.001)
    
    # Get input shape
    try:
        result = preprocess_data()
        if result is not None and len(result) == 7:
            X_train, _, _, _, _, _, _ = result
            if X_train is not None and len(X_train) > 0:
                input_shape = (X_train.shape[1], X_train.shape[2])
            else:
                input_shape = (12, 50)
        else:
            input_shape = (12, 50)
    except:
        input_shape = (12, 50)
    
    model = build_bidirectional_model(
        input_shape=input_shape,
        lstm_units=lstm_units,
        attention_type=attention_type,
        learning_rate=learning_rate
    )
    
    return model


def build_tunable_weather_model(hp):
    """Model builder for weather-only model"""
    
    learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.0005, 0.001, 0.002], default=0.001)
    
    # Get input shape
    try:
        result = preprocess_data()
        if result is not None and len(result) == 7:
            X_train, _, _, _, _, _, _ = result
            if X_train is not None and len(X_train) > 0:
                input_shape = (X_train.shape[1], X_train.shape[2])
            else:
                input_shape = (12, 50)
        else:
            input_shape = (12, 50)
    except:
        input_shape = (12, 50)
    
    model = build_weather_only_model(
        input_shape=input_shape,
        learning_rate=learning_rate
    )
    
    return model


def run_hyperparameter_tuning(model_type='advanced', max_trials=10, epochs=20, 
                             objective='val_loss', directory='models'):
    """
    Run Keras Tuner hyperparameter search
    
    Parameters:
    -----------
    model_type : str
        Type of model to tune: 'advanced', 'bidirectional', 'weather'
    max_trials : int
        Maximum number of trials
    epochs : int
        Number of epochs per trial
    objective : str
        Objective to optimize ('val_loss', 'val_mae', 'val_r2_score')
    directory : str
        Directory to save tuning results
    
    Returns:
    --------
    best_model : tf.keras.Model
        Best model found
    best_hps : kt.HyperParameters
        Best hyperparameters
    """
    
    print("="*60)
    print("🔧 HYPERPARAMETER TUNING")
    print("="*60)
    print(f"Model Type: {model_type.upper()}")
    print(f"Max Trials: {max_trials}")
    print(f"Epochs per trial: {epochs}")
    print(f"Objective: {objective}")
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    try:
        result = preprocess_data()
        if result is None or len(result) != 7:
            print("❌ Preprocessing failed or returned invalid data!")
            print("Creating dummy data for testing...")
            n_samples = 1000
            timesteps = 12
            n_features = 50
            X_train = np.random.random((n_samples, timesteps, n_features))
            X_test = np.random.random((200, timesteps, n_features))
            y_train = np.random.random(n_samples)
            y_test = np.random.random(200)
            scaler_X = None
            scaler_y = None
            features = [f'feature_{i}' for i in range(n_features)]
        else:
            X_train, X_test, y_train, y_test, scaler_X, scaler_y, features = result
            
            # Check if data is valid
            if X_train is None or len(X_train) == 0:
                print("❌ No training data available!")
                return None, None
                
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Create dummy data for testing
        print("⚠️ Creating dummy data for testing...")
        n_samples = 1000
        timesteps = 12
        n_features = 50
        X_train = np.random.random((n_samples, timesteps, n_features))
        X_test = np.random.random((200, timesteps, n_features))
        y_train = np.random.random(n_samples)
        y_test = np.random.random(200)
        scaler_X = None
        scaler_y = None
        features = [f'feature_{i}' for i in range(n_features)]
    
    print(f"\n📈 Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Select model builder
    if model_type == 'advanced':
        build_model_fn = build_tunable_advanced_model
    elif model_type == 'bidirectional':
        build_model_fn = build_tunable_bidirectional_model
    elif model_type == 'weather':
        build_model_fn = build_tunable_weather_model
    else:
        print(f"❌ Unknown model type: {model_type}")
        return None, None
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Initialize tuner
    print("\n🔧 Initializing Keras Tuner...")
    try:
        tuner = kt.RandomSearch(
            build_model_fn,
            objective=kt.Objective(objective, direction='min'),
            max_trials=max_trials,
            executions_per_trial=1,
            directory=directory,
            project_name=f'crop_yield_tuning_{model_type}',
            overwrite=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize tuner: {e}")
        return None, None
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,  # Reduced for faster tuning
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Search for best hyperparameters
    print("\n🚀 Starting hyperparameter search...")
    try:
        tuner.search(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
    except Exception as e:
        print(f"❌ Tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Get best hyperparameters
    try:
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    except Exception as e:
        print(f"❌ Could not retrieve best hyperparameters: {e}")
        return None, None
    
    # Save best hyperparameters
    best_params = {}
    for param in best_hps.values.keys():
        try:
            best_params[param] = best_hps.get(param)
        except:
            pass
    
    # Add model type to params
    best_params['model_type'] = model_type
    best_params['input_shape'] = list(X_train.shape[1:])
    
    with open(f'{directory}/tuner_results_{model_type}.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\n✅ Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Get best model
    print("\n📊 Building best model...")
    try:
        best_model = tuner.get_best_models(num_models=1)[0]
    except Exception as e:
        print(f"❌ Could not retrieve best model: {e}")
        return None, None
    
    # Evaluate best model
    print("\n📊 Evaluating best model...")
    try:
        test_results = best_model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n📈 Test Results:")
        metric_names = best_model.metrics_names
        for i, result in enumerate(test_results):
            print(f"  {metric_names[i]}: {result:.4f}")
    except Exception as e:
        print(f"⚠️ Could not evaluate model: {e}")
    
    # Save best model
    best_model.save(f'{directory}/best_tuned_model_{model_type}.h5')
    print(f"\n✅ Best model saved to {directory}/best_tuned_model_{model_type}.h5")
    
    return best_model, best_hps


def compare_tuning_results():
    """Compare tuning results from different model types"""
    
    print("\n" + "="*60)
    print("📊 COMPARING TUNING RESULTS")
    print("="*60)
    
    model_types = ['advanced', 'bidirectional', 'weather']
    results = {}
    
    for model_type in model_types:
        try:
            with open(f'models/tuner_results_{model_type}.json', 'r') as f:
                results[model_type] = json.load(f)
            print(f"\n✅ Loaded results for {model_type.upper()}")
        except FileNotFoundError:
            print(f"\n⚠️ No results found for {model_type.upper()}")
    
    if results:
        print("\n📊 Best Configurations:")
        for model_type, params in results.items():
            print(f"\n{model_type.upper()}:")
            for param, value in params.items():
                if param not in ['model_type', 'input_shape']:
                    print(f"  {param}: {value}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Check if keras_tuner is available
    try:
        import keras_tuner
        print(f"✅ Keras Tuner version: {keras_tuner.__version__}")
    except ImportError:
        print("❌ Please install keras-tuner first: pip install keras-tuner")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Crop Yield Prediction')
    parser.add_argument('--model', type=str, default='advanced',
                       choices=['advanced', 'bidirectional', 'weather', 'all'],
                       help='Model type to tune')
    parser.add_argument('--trials', type=int, default=10,
                       help='Maximum number of trials (default: 10)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs per trial (default: 20)')
    parser.add_argument('--objective', type=str, default='val_loss',
                       choices=['val_loss', 'val_mae'],
                       help='Objective to optimize')
    parser.add_argument('--compare', action='store_true',
                       help='Compare results from different model types')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare results from all model types
        compare_tuning_results()
    elif args.model == 'all':
        # Tune all model types
        for model_type in ['advanced', 'bidirectional', 'weather']:
            print(f"\n{'='*60}")
            print(f"Tuning {model_type.upper()} model...")
            print(f"{'='*60}")
            run_hyperparameter_tuning(
                model_type=model_type,
                max_trials=args.trials,
                epochs=args.epochs,
                objective=args.objective
            )
        
        # Compare results
        compare_tuning_results()
    else:
        # Tune single model
        best_model, best_hps = run_hyperparameter_tuning(
            model_type=args.model,
            max_trials=args.trials,
            epochs=args.epochs,
            objective=args.objective
        )
        
        if best_model is not None:
            print("\n🎉 Hyperparameter tuning completed successfully!")
            print("\n💡 To use the best hyperparameters, update your model configuration with:")
            print("  " + json.dumps(best_hps.values, indent=2))