"""
Advanced LSTM Models with Attention Mechanisms for Crop Yield Prediction
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.attention import Attention, MultiHeadAttention, TemporalAttention, FeatureAttention

def build_advanced_model(input_shape, 
                         lstm_units=[128, 64, 32],
                         dropout_rates=[0.3, 0.3, 0.2],
                         attention_type='multihead',
                         use_batch_norm=True,
                         learning_rate=0.001,
                         l2_reg=0.001,
                         num_heads=4):
    """
    Build advanced LSTM model with attention for weather-crop prediction
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    lstm_units : list
        Number of units in each LSTM layer
    dropout_rates : list
        Dropout rate for each LSTM layer
    attention_type : str
        Type of attention to use: 'none', 'single', 'multihead', 'temporal', 'feature'
    use_batch_norm : bool
        Whether to use batch normalization
    learning_rate : float
        Learning rate for optimizer
    l2_reg : float
        L2 regularization factor
    num_heads : int
        Number of heads for multi-head attention (must divide LSTM units)
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    
    # Validate input dimensions for multi-head attention
    if attention_type == 'multihead' and lstm_units[-1] % num_heads != 0:
        print(f"⚠️ Warning: LSTM units ({lstm_units[-1]}) not divisible by num_heads ({num_heads})")
        print(f"Adjusting num_heads to {lstm_units[-1]}")
        num_heads = lstm_units[-1] // 2
        while lstm_units[-1] % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        print(f"Using num_heads={num_heads}")
    
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Add LSTM layers with regularization
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1) or (attention_type != 'none')
        
        model.add(layers.LSTM(units, 
                              return_sequences=return_sequences,
                              dropout=dropout_rates[i] if i < len(dropout_rates) else 0.2,
                              recurrent_dropout=dropout_rates[i] if i < len(dropout_rates) else 0.2,
                              kernel_regularizer=regularizers.l2(l2_reg),
                              recurrent_regularizer=regularizers.l2(l2_reg),
                              name=f'lstm_{i}'))
        
        if use_batch_norm and return_sequences:
            model.add(layers.BatchNormalization(name=f'bn_{i}'))
    
    # Attention mechanism
    if attention_type == 'single':
        model.add(Attention(name='attention_single'))
        
    elif attention_type == 'multihead':
        model.add(MultiHeadAttention(num_heads=num_heads, name='attention_multihead'))
        # After multi-head attention, we need to reduce dimensions
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
        
    elif attention_type == 'temporal':
        model.add(TemporalAttention(name='attention_temporal'))
        
    elif attention_type == 'feature':
        model.add(FeatureAttention(name='attention_feature'))
        # After feature attention, pool over timesteps
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
        
    # If no attention, ensure we have a single output from LSTM
    elif attention_type == 'none':
        if model.layers[-1].return_sequences:
            # Take last timestep output
            model.add(layers.Lambda(lambda x: x[:, -1, :], name='last_timestep'))
    
    # Dense layers for prediction
    model.add(layers.Dense(128, 
                          activation='relu', 
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name='dense_1'))
    model.add(layers.Dropout(0.3, name='dropout_1'))
    if use_batch_norm:
        model.add(layers.BatchNormalization(name='bn_dense_1'))
    
    model.add(layers.Dense(64, 
                          activation='relu', 
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name='dense_2'))
    model.add(layers.Dropout(0.2, name='dropout_2'))
    if use_batch_norm:
        model.add(layers.BatchNormalization(name='bn_dense_2'))
    
    model.add(layers.Dense(32, activation='relu', name='dense_3'))
    model.add(layers.Dropout(0.1, name='dropout_3'))
    
    # Output layer for regression (crop yield)
    model.add(layers.Dense(1, activation='linear', name='output'))
    
    # Compile with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    # Use appropriate metrics (R2Score requires newer TF version)
    try:
        # For TensorFlow 2.8+
        model.compile(
            optimizer=optimizer, 
            loss='mse', 
            metrics=['mae', 'mse', tf.keras.metrics.R2Score(name='r2_score')]
        )
    except:
        # For older TensorFlow versions
        model.compile(
            optimizer=optimizer, 
            loss='mse', 
            metrics=['mae', 'mse']
        )
    
    return model


def build_weather_only_model(input_shape, learning_rate=0.001):
    """
    Simpler model focusing only on weather patterns
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True, dropout=0.2, name='lstm_1'),
        layers.LSTM(32, dropout=0.2, name='lstm_2'),
        layers.Dense(16, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout'),
        layers.Dense(1, activation='linear', name='output')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def build_bidirectional_model(input_shape, 
                             lstm_units=64,
                             attention_type='single',
                             learning_rate=0.001):
    """
    Build bidirectional LSTM model with attention
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    lstm_units : int
        Number of LSTM units
    attention_type : str
        Type of attention to use: 'none', 'single', 'multihead'
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Bidirectional LSTM
    model.add(layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2),
        name='bidirectional_lstm'
    ))
    
    # Attention mechanism
    if attention_type == 'single':
        model.add(Attention(name='attention'))
    elif attention_type == 'multihead':
        num_heads = 4
        model.add(MultiHeadAttention(num_heads=num_heads, name='attention_multihead'))
        model.add(layers.GlobalAveragePooling1D(name='global_avg_pool'))
    else:
        # Take last timestep
        model.add(layers.Lambda(lambda x: x[:, -1, :], name='last_timestep'))
    
    # Dense layers
    model.add(layers.Dense(32, activation='relu', name='dense_1'))
    model.add(layers.Dropout(0.2, name='dropout'))
    model.add(layers.Dense(1, activation='linear', name='output'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def build_conv_lstm_model(input_shape, learning_rate=0.001):
    """
    Build ConvLSTM model for spatial-temporal patterns
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    # Reshape input for Conv1D (add channel dimension)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((input_shape[0], input_shape[1], 1), name='reshape'),
        
        # Conv1D layers
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1'),
        layers.BatchNormalization(name='bn_conv_1'),
        layers.MaxPooling2D((2, 1), name='maxpool_1'),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        layers.BatchNormalization(name='bn_conv_2'),
        layers.MaxPooling2D((2, 1), name='maxpool_2'),
        
        # Flatten and reshape for LSTM
        layers.Reshape((input_shape[0], -1), name='reshape_back'),
        
        # LSTM layers
        layers.LSTM(64, return_sequences=True, dropout=0.2, name='lstm_1'),
        Attention(name='attention'),
        
        # Dense layers
        layers.Dense(32, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout'),
        layers.Dense(1, activation='linear', name='output')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def get_model_summary(input_shape, model_type='advanced'):
    """
    Print model architecture summary
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    model_type : str
        Type of model to summarize: 'advanced', 'weather', 'bidirectional', 'conv_lstm'
    """
    if model_type == 'advanced':
        model = build_advanced_model(input_shape)
        print("="*60)
        print("Advanced Model Architecture")
        print("="*60)
    elif model_type == 'weather':
        model = build_weather_only_model(input_shape)
        print("="*60)
        print("Weather Only Model Architecture")
        print("="*60)
    elif model_type == 'bidirectional':
        model = build_bidirectional_model(input_shape)
        print("="*60)
        print("Bidirectional LSTM Model Architecture")
        print("="*60)
    elif model_type == 'conv_lstm':
        model = build_conv_lstm_model(input_shape)
        print("="*60)
        print("ConvLSTM Model Architecture")
        print("="*60)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.summary()
    return model


def compare_attention_types(input_shape):
    """
    Compare different attention mechanisms
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    """
    attention_types = ['none', 'single', 'multihead', 'temporal', 'feature']
    
    print("="*60)
    print("Comparing Different Attention Mechanisms")
    print("="*60)
    print(f"Input shape: {input_shape}\n")
    
    models_dict = {}
    
    for att_type in attention_types:
        print(f"\n📊 Building model with {att_type} attention...")
        try:
            model = build_advanced_model(
                input_shape=input_shape,
                attention_type=att_type,
                lstm_units=[64, 32],  # Smaller for quick testing
                use_batch_norm=False
            )
            models_dict[att_type] = model
            print(f"✅ {att_type.upper()} attention model created successfully")
            
            # Count parameters
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"❌ Failed to build {att_type} attention model: {e}")
    
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    
    for att_type, model in models_dict.items():
        print(f"\n{att_type.upper()} Attention:")
        print(f"  - Total params: {model.count_params():,}")
        print(f"  - Output shape: {model.output_shape}")
    
    return models_dict


if __name__ == "__main__":
    # Test with dummy input
    timesteps = 12  # 12 months
    n_features = 50  # Weather + soil + crop features
    dummy_input = (timesteps, n_features)
    
    print("="*60)
    print("🌾 CROP YIELD PREDICTION MODELS")
    print("="*60)
    
    # Test 1: Build advanced model with different attention types
    print("\n📊 Test 1: Building Advanced Model with Multi-Head Attention")
    try:
        model1 = build_advanced_model(dummy_input, attention_type='multihead', num_heads=5)
        print("✅ Advanced model created successfully!")
        print(f"   Input shape: {model1.input_shape}")
        print(f"   Output shape: {model1.output_shape}")
        print(f"   Total parameters: {model1.count_params():,}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Build weather-only model
    print("\n📊 Test 2: Building Weather-Only Model")
    try:
        model2 = build_weather_only_model(dummy_input)
        print("✅ Weather-only model created successfully!")
        print(f"   Input shape: {model2.input_shape}")
        print(f"   Output shape: {model2.output_shape}")
        print(f"   Total parameters: {model2.count_params():,}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Build bidirectional model
    print("\n📊 Test 3: Building Bidirectional Model")
    try:
        model3 = build_bidirectional_model(dummy_input, attention_type='single')
        print("✅ Bidirectional model created successfully!")
        print(f"   Input shape: {model3.input_shape}")
        print(f"   Output shape: {model3.output_shape}")
        print(f"   Total parameters: {model3.count_params():,}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Build ConvLSTM model
    print("\n📊 Test 4: Building ConvLSTM Model")
    try:
        model4 = build_conv_lstm_model(dummy_input)
        print("✅ ConvLSTM model created successfully!")
        print(f"   Input shape: {model4.input_shape}")
        print(f"   Output shape: {model4.output_shape}")
        print(f"   Total parameters: {model4.count_params():,}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Compare attention mechanisms
    print("\n📊 Test 5: Comparing Attention Mechanisms")
    try:
        models_dict = compare_attention_types(dummy_input)
        print(f"\n✅ Successfully compared {len(models_dict)} attention mechanisms")
    except Exception as e:
        print(f"❌ Error comparing attention mechanisms: {e}")
    
    # Test 6: Test model with random data
    print("\n📊 Test 6: Testing Forward Pass")
    try:
        test_input = tf.random.normal((32, timesteps, n_features))
        predictions = model1(test_input)
        print(f"✅ Forward pass successful!")
        print(f"   Input batch shape: {test_input.shape}")
        print(f"   Output predictions shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[:3].numpy().flatten()}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    print("\n" + "🎉"*30)
    print("All models are ready for training!")
    print("🎉"*30)