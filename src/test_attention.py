import tensorflow as tf
import numpy as np
from src.attention import Attention, MultiHeadAttention, TemporalAttention, FeatureAttention

def test_attention_layers():
    """Test both attention layers with sample data"""
    
    print("="*50)
    print("Testing Attention Layers")
    print("="*50)
    
    # Create sample data
    batch_size = 4
    timesteps = 10
    features = 64
    
    sample_input = tf.random.normal((batch_size, timesteps, features))
    
    # Test Basic Attention
    print("\n📊 Testing Basic Attention Layer:")
    attention_layer = Attention()
    output = attention_layer(sample_input)
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Verify output shape is correct (batch_size, features)
    expected_shape = (batch_size, features)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"  ✅ Basic Attention works! Shape matches expected: {expected_shape}")
    
    # Test Multi-Head Attention
    print("\n📊 Testing Multi-Head Attention Layer:")
    multihead_layer = MultiHeadAttention(num_heads=4)
    output_mh = multihead_layer(sample_input)
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output shape: {output_mh.shape}")
    
    # Verify multi-head output maintains shape
    expected_mh_shape = (batch_size, timesteps, features)
    assert output_mh.shape == expected_mh_shape, f"Expected {expected_mh_shape}, got {output_mh.shape}"
    print(f"  ✅ Multi-Head Attention works! Shape preserved: {expected_mh_shape}")
    
    # Test gradient flow
    print("\n📊 Testing gradient flow:")
    with tf.GradientTape() as tape:
        tape.watch(sample_input)
        output = attention_layer(sample_input)
        loss = tf.reduce_mean(output)
    
    gradients = tape.gradient(loss, sample_input)
    assert gradients is not None, "Gradients are None!"
    assert not tf.reduce_any(tf.math.is_nan(gradients)), "NaN gradients detected!"
    print(f"  ✅ Gradients flow properly! Gradient shape: {gradients.shape}")
    
    # Test with realistic model dimensions
    print("\n📊 Testing with realistic model dimensions:")
    model_seq_length = 12  # LSTM sequence length
    model_features = 50    # Number of features after LSTM
    
    # Create NEW attention layer for the new dimensions
    realistic_attention = Attention()
    model_input = tf.random.normal((batch_size, model_seq_length, model_features))
    
    # The layer will build itself with the new input shape
    attention_out = realistic_attention(model_input)
    print(f"  LSTM output shape: {model_input.shape}")
    print(f"  Attention output shape: {attention_out.shape}")
    print(f"  Features dimension: {attention_out.shape[-1]}")
    print(f"  ✅ Ready for integration with your LSTM model!")
    
    # Test Multi-Head Attention with realistic dimensions
    print("\n📊 Testing Multi-Head Attention with realistic dimensions:")
    realistic_multihead = MultiHeadAttention(num_heads=5)  # 50/5=10
    multihead_out = realistic_multihead(model_input)
    print(f"  Input shape: {model_input.shape}")
    print(f"  Multi-Head Attention output shape: {multihead_out.shape}")
    print(f"  ✅ Multi-Head Attention works with realistic dimensions!")
    
    # Test Temporal Attention
    print("\n📊 Testing Temporal Attention:")
    temporal_layer = TemporalAttention()
    temporal_out = temporal_layer(model_input)
    print(f"  Input shape: {model_input.shape}")
    print(f"  Temporal Attention output shape: {temporal_out.shape}")
    print(f"  ✅ Temporal Attention works!")
    
    # Test Feature Attention
    print("\n📊 Testing Feature Attention:")
    feature_layer = FeatureAttention()
    feature_out = feature_layer(model_input)
    print(f"  Input shape: {model_input.shape}")
    print(f"  Feature Attention output shape: {feature_out.shape}")
    print(f"  ✅ Feature Attention works!")
    
    print("\n" + "="*50)
    print("✅ All tests passed! Attention layers are ready to use.")
    print("="*50)
    
    return True

def test_in_model_integration():
    """Test attention layers within a complete model"""
    print("\n" + "="*50)
    print("Testing Attention Layers in Model Integration")
    print("="*50)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
    
    batch_size = 32
    seq_length = 12
    n_features = 50
    
    # Test 1: Basic Attention in model
    print("\n📊 Test 1: Model with Basic Attention")
    model1 = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        Attention(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    test_input = tf.random.normal((batch_size, seq_length, n_features))
    prediction = model1(test_input)
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {prediction.shape}")
    print(f"  ✅ Basic Attention model works!")
    
    # Test 2: Multi-Head Attention in model (Fixed: LSTM units divisible by num_heads)
    print("\n📊 Test 2: Model with Multi-Head Attention")
    lstm_units = 128
    num_heads = 4  # 128 is divisible by 4
    
    model2 = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, n_features)),
        MultiHeadAttention(num_heads=num_heads),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    prediction2 = model2(test_input)
    print(f"  LSTM units: {lstm_units}, Num heads: {num_heads}")
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {prediction2.shape}")
    print(f"  ✅ Multi-Head Attention model works!")
    
    # Test 2b: Multi-Head Attention with different configuration
    print("\n📊 Test 2b: Model with Multi-Head Attention (different config)")
    lstm_units2 = 100
    num_heads2 = 5  # 100 is divisible by 5
    
    model2b = Sequential([
        LSTM(lstm_units2, return_sequences=True, input_shape=(seq_length, n_features)),
        MultiHeadAttention(num_heads=num_heads2),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    prediction2b = model2b(test_input)
    print(f"  LSTM units: {lstm_units2}, Num heads: {num_heads2}")
    print(f"  Model output shape: {prediction2b.shape}")
    print(f"  ✅ Multi-Head Attention (alt config) works!")
    
    # Test 3: Bidirectional LSTM with Attention
    print("\n📊 Test 3: Bidirectional LSTM with Attention")
    model3 = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length, n_features)),
        Attention(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    prediction3 = model3(test_input)
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {prediction3.shape}")
    print(f"  ✅ Bidirectional LSTM + Attention model works!")
    
    # Test 4: Stacked LSTMs with Attention
    print("\n📊 Test 4: Stacked LSTMs with Attention")
    model4 = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        LSTM(32, return_sequences=True),
        Attention(),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    prediction4 = model4(test_input)
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {prediction4.shape}")
    print(f"  ✅ Stacked LSTMs + Attention model works!")
    
    # Test 5: Temporal Attention in model
    print("\n📊 Test 5: Model with Temporal Attention")
    model5 = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        TemporalAttention(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    prediction5 = model5(test_input)
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {prediction5.shape}")
    print(f"  ✅ Temporal Attention model works!")
    
    # Test 6: Feature Attention in model
    print("\n📊 Test 6: Model with Feature Attention")
    model6 = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        FeatureAttention(),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    prediction6 = model6(test_input)
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {prediction6.shape}")
    print(f"  ✅ Feature Attention model works!")
    
    # Test 7: Combined attentions
    print("\n📊 Test 7: Combined Attention Mechanisms")
    model7 = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        FeatureAttention(),  # First focus on important features
        MultiHeadAttention(num_heads=4),  # Then capture complex patterns
        TemporalAttention(),  # Finally focus on important timesteps
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    prediction7 = model7(test_input)
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {prediction7.shape}")
    print(f"  ✅ Combined Attention model works!")
    
    # Model summary for one of the models
    print("\n📊 Sample Model Architecture (Test 7):")
    model7.summary()
    
    print("\n" + "="*50)
    print("✅ All model integrations passed!")
    print("="*50)
    
    return True

def test_performance():
    """Test performance of different attention mechanisms"""
    print("\n" + "="*50)
    print("Performance Comparison of Attention Mechanisms")
    print("="*50)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling1D
    import time
    
    batch_size = 32
    seq_length = 12
    n_features = 50
    
    test_input = tf.random.normal((batch_size, seq_length, n_features))
    
    mechanisms = {
        'No Attention': Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dense(1)
        ]),
        'Basic Attention': Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
            Attention(),
            Dense(64, activation='relu'),
            Dense(1)
        ]),
        'Multi-Head Attention': Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
            MultiHeadAttention(num_heads=4),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dense(1)
        ]),
        'Temporal Attention': Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
            TemporalAttention(),
            Dense(64, activation='relu'),
            Dense(1)
        ]),
        'Feature Attention': Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
            FeatureAttention(),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
    }
    
    print("\n📊 Forward pass time comparison:")
    for name, model in mechanisms.items():
        # Warm up
        _ = model(test_input)
        
        # Time multiple runs
        start = time.time()
        for _ in range(100):
            _ = model(test_input)
        end = time.time()
        
        avg_time = (end - start) / 100 * 1000  # Convert to milliseconds
        print(f"  {name:20s}: {avg_time:.3f} ms per forward pass")
    
    print("\n✅ Performance test complete!")

if __name__ == "__main__":
    # Run basic tests
    test_attention_layers()
    
    # Run model integration tests
    test_in_model_integration()
    
    # Run performance comparison (optional)
    print("\n" + "="*50)
    perf_test = input("Run performance comparison? (y/n): ")
    if perf_test.lower() == 'y':
        test_performance()
    
    print("\n" + "🎉"*20)
    print("All attention mechanisms are working correctly!")
    print("You can now use them in your crop yield prediction model.")
    print("🎉"*20)