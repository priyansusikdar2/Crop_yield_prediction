import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from src.preprocess import preprocess_data
from src.model import build_advanced_model, build_bidirectional_model, build_weather_only_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def plot_predictions(y_test_real, y_pred_real, save_path='models/predictions_plot.png'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_test_real, y_pred_real, alpha=0.5, label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_test_real.min(), y_pred_real.min())
    max_val = max(y_test_real.max(), y_pred_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Yield (kg/ha)', fontsize=12)
    plt.ylabel('Predicted Yield (kg/ha)', fontsize=12)
    plt.title('Actual vs Predicted Crop Yield', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_test_real, y_pred_real)
    plt.text(0.05, 0.95, f'R² Score: {r2:.4f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    return r2

def plot_residuals(y_test_real, y_pred_real, save_path='models/residuals_plot.png'):
    """Plot residuals distribution"""
    residuals = y_test_real - y_pred_real
    
    plt.figure(figsize=(12, 5))
    
    # Residuals scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred_real, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Yield (kg/ha)', fontsize=12)
    plt.ylabel('Residuals (kg/ha)', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Residuals histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals (kg/ha)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residuals Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    return residuals

def plot_feature_importance(model, feature_names, save_path='models/feature_importance.png'):
    """Plot feature importance based on attention weights"""
    try:
        # Get attention layer weights if available
        attention_layer = None
        for layer in model.layers:
            if 'attention' in layer.name:
                attention_layer = layer
                break
        
        if attention_layer and hasattr(attention_layer, 'attention_weights'):
            # Extract attention weights
            weights = attention_layer.attention_weights.numpy()
            
            if len(weights.shape) == 2:
                # For basic attention
                importance = np.mean(weights, axis=0)
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(importance)), importance)
                plt.xlabel('Feature Index', fontsize=12)
                plt.ylabel('Attention Weight', fontsize=12)
                plt.title('Feature Importance based on Attention Weights', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.savefig(save_path, dpi=150)
                plt.show()
            else:
                print("  Cannot visualize multi-head attention weights easily")
        else:
            print("  No attention layer found or cannot extract weights")
    except Exception as e:
        print(f"  Could not plot feature importance: {e}")

def train_model(model_type='advanced', model_config=None):
    """
    Main training function
    
    Parameters:
    -----------
    model_type : str
        Type of model to train: 'advanced', 'bidirectional', 'weather'
    model_config : dict
        Configuration for the model
    """
    
    print("="*60)
    print("🌾 CROP YIELD PREDICTION - TRAINING")
    print("="*60)
    print(f"Model Type: {model_type.upper()}")
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    result = preprocess_data()
    
    # Handle different return values from preprocess_data
    if len(result) == 7:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, features = result
    else:
        print("❌ Preprocessing failed!")
        return None, None
    
    print(f"\n📈 Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Features: {len(features)}")
    
    # Check if we have enough data
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ No training data available!")
        return None, None
    
    # Build model based on type
    print("\n🏗️ Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Default configuration
    default_config = {
        'lstm_units': [128, 64, 32],
        'dropout_rates': [0.3, 0.3, 0.2],
        'attention_type': 'multihead',
        'use_batch_norm': True,
        'learning_rate': 0.001,
        'l2_reg': 0.001,
        'num_heads': 4
    }
    
    # Override with user config
    if model_config:
        default_config.update(model_config)
    
    if model_type == 'advanced':
        model = build_advanced_model(
            input_shape=input_shape,
            lstm_units=default_config['lstm_units'],
            dropout_rates=default_config['dropout_rates'],
            attention_type=default_config['attention_type'],
            use_batch_norm=default_config['use_batch_norm'],
            learning_rate=default_config['learning_rate'],
            l2_reg=default_config['l2_reg'],
            num_heads=default_config['num_heads']
        )
    elif model_type == 'bidirectional':
        model = build_bidirectional_model(
            input_shape=input_shape,
            lstm_units=default_config['lstm_units'][0],
            attention_type=default_config['attention_type'],
            learning_rate=default_config['learning_rate']
        )
    elif model_type == 'weather':
        model = build_weather_only_model(
            input_shape=input_shape,
            learning_rate=default_config['learning_rate']
        )
    else:
        print(f"❌ Unknown model type: {model_type}")
        return None, None
    
    model.summary()
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True, 
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=10, 
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'models/best_model_{model_type}.h5', 
            monitor='val_loss',
            save_best_only=True, 
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Train
    print("\n🚀 Starting training...")
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None
    
    # Save final model
    model.save(f'models/final_model_{model_type}.h5')
    print(f"\n✅ Model saved to models/final_model_{model_type}.h5")
    
    # Save scalers and feature info
    joblib.dump(scaler_X, f'models/scaler_X_{model_type}.pkl')
    joblib.dump(scaler_y, f'models/scaler_y_{model_type}.pkl')
    joblib.dump(features, f'models/features_{model_type}.pkl')
    
    # Evaluate
    print("\n📊 Evaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📈 Test Results (normalized scale):")
    metric_names = model.metrics_names
    for i, result in enumerate(test_results):
        print(f"  {metric_names[i]}: {result:.4f}")
    
    # Inverse transform for real metrics
    print("\n📊 Making predictions...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Ensure arrays are 1D
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Calculate real-world metrics
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2_real = r2_score(y_test_real, y_pred_real)
    
    # Calculate MAPE (avoid division by zero)
    y_test_nonzero = y_test_real.copy()
    y_test_nonzero[y_test_nonzero == 0] = 1
    mape_real = np.mean(np.abs((y_test_real - y_pred_real) / y_test_nonzero)) * 100
    
    print(f"\n🌾 Real-world metrics:")
    print(f"  MAE: {mae_real:.2f} kg/ha")
    print(f"  RMSE: {rmse_real:.2f} kg/ha")
    print(f"  R² Score: {r2_real:.4f}")
    print(f"  MAPE: {mape_real:.2f}%")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    if 'mae' in history.history:
        axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[0, 1].set_title('Model MAE', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # R² plot if available
    if 'r2_score' in history.history:
        axes[1, 0].plot(history.history['r2_score'], label='Train R²', linewidth=2)
        axes[1, 0].plot(history.history['val_r2_score'], label='Validation R²', linewidth=2)
        axes[1, 0].set_title('Model R² Score', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    # Learning rate plot
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'models/training_history_{model_type}.png', dpi=150)
    plt.show()
    
    # Plot predictions
    print("\n📊 Plotting predictions...")
    r2 = plot_predictions(y_test_real, y_pred_real, f'models/predictions_plot_{model_type}.png')
    
    # Plot residuals
    print("\n📊 Plotting residuals...")
    residuals = plot_residuals(y_test_real, y_pred_real, f'models/residuals_plot_{model_type}.png')
    
    # Plot feature importance (if possible)
    print("\n📊 Analyzing feature importance...")
    plot_feature_importance(model, features, f'models/feature_importance_{model_type}.png')
    
    # Save metrics to file
    metrics = {
        'model_type': model_type,
        'mae_normalized': test_results[1] if len(test_results) > 1 else None,
        'mse_normalized': test_results[2] if len(test_results) > 2 else None,
        'mae_real': float(mae_real),
        'rmse_real': float(rmse_real),
        'r2_real': float(r2_real),
        'mape_real': float(mape_real),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(features),
        'input_shape': input_shape
    }
    
    # Save metrics
    with open(f'models/metrics_{model_type}.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Save as JSON
    import json
    with open(f'models/metrics_{model_type}.json', 'w') as f:
        # Convert numpy values to Python types
        metrics_serializable = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)
    
    print("\n✅ Metrics saved to models/")
    print(f"\n✅ Training completed successfully!")
    return model, history

def compare_models():
    """Train and compare multiple model architectures"""
    
    print("="*60)
    print("📊 COMPARING DIFFERENT MODEL ARCHITECTURES")
    print("="*60)
    
    model_types = ['advanced', 'bidirectional', 'weather']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*60}")
        
        model, history = train_model(model_type=model_type)
        
        if model is not None:
            # Load metrics
            import json
            with open(f'models/metrics_{model_type}.json', 'r') as f:
                results[model_type] = json.load(f)
    
    # Compare results
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df[['mae_real', 'rmse_real', 'r2_real', 'mape_real']].to_string())
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = list(results.keys())
    mae_values = [results[m]['mae_real'] for m in models]
    rmse_values = [results[m]['rmse_real'] for m in models]
    r2_values = [results[m]['r2_real'] for m in models]
    mape_values = [results[m]['mape_real'] for m in models]
    
    axes[0, 0].bar(models, mae_values, color=['blue', 'green', 'orange'])
    axes[0, 0].set_title('MAE Comparison (kg/ha)', fontsize=12)
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(models, rmse_values, color=['blue', 'green', 'orange'])
    axes[0, 1].set_title('RMSE Comparison (kg/ha)', fontsize=12)
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].bar(models, r2_values, color=['blue', 'green', 'orange'])
    axes[1, 0].set_title('R² Score Comparison', fontsize=12)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(models, mape_values, color=['blue', 'green', 'orange'])
    axes[1, 1].set_title('MAPE Comparison (%)', fontsize=12)
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=150)
    plt.show()
    
    # Determine best model
    best_model = min(results.items(), key=lambda x: x[1]['rmse_real'])[0]
    print(f"\n🏆 Best model based on RMSE: {best_model.upper()}")
    print(f"   RMSE: {results[best_model]['rmse_real']:.2f} kg/ha")
    print(f"   MAE: {results[best_model]['mae_real']:.2f} kg/ha")
    print(f"   R²: {results[best_model]['r2_real']:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Crop Yield Prediction Model')
    parser.add_argument('--model', type=str, default='advanced', 
                       choices=['advanced', 'bidirectional', 'weather', 'compare'],
                       help='Model type to train')
    parser.add_argument('--attention', type=str, default='multihead',
                       choices=['none', 'single', 'multihead', 'temporal', 'feature'],
                       help='Attention type for advanced model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    if args.model == 'compare':
        # Train and compare all models
        compare_models()
    else:
        # Train single model
        config = {
            'attention_type': args.attention,
            'learning_rate': args.lr
        }
        
        model, history = train_model(
            model_type=args.model,
            model_config=config
        )
        
        if model is not None:
            print("\n🎉 Model training completed successfully!")
        else:
            print("\n❌ Model training failed!")