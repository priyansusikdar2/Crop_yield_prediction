"""
Attention Mechanisms for Crop Yield Prediction
Custom attention layers for LSTM-based crop yield prediction models
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class Attention(Layer):
    """Custom Attention Layer for LSTM
    
    This layer computes attention weights over timesteps and returns
    a context vector that summarizes the most important information.
    
    Input shape: (batch_size, timesteps, features)
    Output shape: (batch_size, features)
    """
    
    def __init__(self, return_attention_weights=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.return_attention_weights = return_attention_weights
    
    def build(self, input_shape):
        # Input shape: (batch_size, timesteps, features)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, timesteps, features)
        # Calculate attention scores
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        # Softmax over timesteps
        attention_weights = tf.nn.softmax(score, axis=1)
        # Apply attention weights to input
        context_vector = attention_weights * x
        # Sum over timesteps
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        if self.return_attention_weights:
            return context_vector, attention_weights
        return context_vector
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'return_attention_weights': self.return_attention_weights})
        return config


class MultiHeadAttention(Layer):
    """Multi-Head Attention Layer for capturing complex patterns
    
    This layer implements multi-head attention mechanism similar to transformers.
    
    Input shape: (batch_size, timesteps, features)
    Output shape: (batch_size, timesteps, features)
    """
    
    def __init__(self, num_heads=4, use_projection=True, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.use_projection = use_projection
    
    def build(self, input_shape):
        self.features = input_shape[-1]
        
        # Ensure features divisible by num_heads
        assert self.features % self.num_heads == 0, \
            f"Features ({self.features}) must be divisible by num_heads ({self.num_heads})"
        
        self.depth = self.features // self.num_heads
        
        # Query, Key, Value weight matrices
        self.W_q = self.add_weight(
            name='W_q',
            shape=(self.features, self.features),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_k = self.add_weight(
            name='W_k',
            shape=(self.features, self.features),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_v = self.add_weight(
            name='W_v',
            shape=(self.features, self.features),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_projection:
            self.W_o = self.add_weight(
                name='W_o',
                shape=(self.features, self.features),
                initializer='glorot_uniform',
                trainable=True
            )
        
        super(MultiHeadAttention, self).build(input_shape)
    
    def split_heads(self, x, batch_size, timesteps):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, timesteps, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, timesteps, depth)
    
    def scaled_dot_product_attention(self, q, k, v):
        """Calculate scaled dot-product attention"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        timesteps = tf.shape(x)[1]
        
        # Linear projections
        x_flat = tf.reshape(x, [-1, self.features])
        Q = tf.matmul(x_flat, self.W_q)
        K = tf.matmul(x_flat, self.W_k)
        V = tf.matmul(x_flat, self.W_v)
        
        # Reshape back to 3D
        Q = tf.reshape(Q, (batch_size, timesteps, self.features))
        K = tf.reshape(K, (batch_size, timesteps, self.features))
        V = tf.reshape(V, (batch_size, timesteps, self.features))
        
        # Split into heads
        Q = self.split_heads(Q, batch_size, timesteps)
        K = self.split_heads(K, batch_size, timesteps)
        V = self.split_heads(V, batch_size, timesteps)
        
        # Apply attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, timesteps, self.features))
        
        # Final projection
        if self.use_projection:
            concat_attention_flat = tf.reshape(concat_attention, [-1, self.features])
            output = tf.matmul(concat_attention_flat, self.W_o)
            output = tf.reshape(output, (batch_size, timesteps, self.features))
        else:
            output = concat_attention
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'use_projection': self.use_projection
        })
        return config


class TemporalAttention(Layer):
    """Temporal Attention for time-series data (FIXED)
    
    This attention mechanism focuses on temporal patterns across sequences.
    
    Input shape: (batch_size, timesteps, features)
    Output shape: (batch_size, features)
    """
    
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.timesteps = input_shape[1]
        self.features = input_shape[2]
        
        # Attention weights for each timestep
        self.attention_weights = self.add_weight(
            name='temporal_weights',
            shape=(self.timesteps, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(TemporalAttention, self).build(input_shape)
    
    def call(self, x):
        # Calculate temporal attention scores
        # x shape: (batch_size, timesteps, features)
        # attention_weights shape: (timesteps, 1)
        
        # Apply softmax to get attention distribution
        alpha = tf.nn.softmax(self.attention_weights, axis=0)  # (timesteps, 1)
        
        # Reshape alpha to (1, timesteps, 1) for broadcasting
        alpha = tf.reshape(alpha, (1, self.timesteps, 1))
        
        # Apply attention weights (broadcasting works)
        weighted_x = x * alpha  # (batch, timesteps, features) * (1, timesteps, 1)
        
        # Sum over timesteps
        output = tf.reduce_sum(weighted_x, axis=1)  # (batch_size, features)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        return config


class FeatureAttention(Layer):
    """Feature Attention for selecting important features
    
    This attention mechanism learns which features are most important
    for prediction at each timestep.
    
    Input shape: (batch_size, timesteps, features)
    Output shape: (batch_size, timesteps, features) - same shape with attention applied
    """
    
    def __init__(self, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.features = input_shape[-1]
        
        # Feature importance weights
        self.feature_weights = self.add_weight(
            name='feature_weights',
            shape=(self.features, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(FeatureAttention, self).build(input_shape)
    
    def call(self, x):
        # Calculate feature attention scores
        # x shape: (batch_size, timesteps, features)
        # feature_weights shape: (features, 1)
        
        # Apply softmax to get attention distribution over features
        alpha = tf.nn.softmax(self.feature_weights, axis=0)  # (features, 1)
        
        # Reshape alpha for broadcasting
        alpha = tf.reshape(alpha, (1, 1, self.features))  # (1, 1, features)
        
        # Apply attention to features
        attended_features = x * alpha  # Broadcasting works
        
        return attended_features
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(FeatureAttention, self).get_config()
        return config


# Testing functions
def test_attention_layers():
    """Test all attention layers with sample data"""
    
    print("="*60)
    print("🧪 TESTING ATTENTION LAYERS")
    print("="*60)
    
    # Test parameters
    batch_size = 4
    timesteps = 10
    features = 64
    
    sample_input = tf.random.normal((batch_size, timesteps, features))
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic Attention Layer
    print("\n📊 Test 1: Basic Attention Layer")
    total_tests += 1
    try:
        attention_layer = Attention()
        output = attention_layer(sample_input)
        expected_shape = (batch_size, features)
        
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"   ✅ Basic Attention works! Shape matches: {expected_shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Basic Attention failed: {e}")
    
    # Test 2: Basic Attention with attention weights
    print("\n📊 Test 2: Basic Attention with weights")
    total_tests += 1
    try:
        attention_layer = Attention(return_attention_weights=True)
        output, weights = attention_layer(sample_input)
        expected_shape = (batch_size, features)
        expected_weights_shape = (batch_size, timesteps, 1)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Weights shape: {weights.shape}")
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert weights.shape == expected_weights_shape, f"Expected {expected_weights_shape}, got {weights.shape}"
        print(f"   ✅ Attention with weights works!")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Attention with weights failed: {e}")
    
    # Test 3: Multi-Head Attention
    print("\n📊 Test 3: Multi-Head Attention Layer")
    total_tests += 1
    try:
        multihead_layer = MultiHeadAttention(num_heads=4)
        output = multihead_layer(sample_input)
        expected_shape = (batch_size, timesteps, features)
        
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"   ✅ Multi-Head Attention works! Shape preserved: {expected_shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Multi-Head Attention failed: {e}")
    
    # Test 4: Temporal Attention (FIXED)
    print("\n📊 Test 4: Temporal Attention Layer")
    total_tests += 1
    try:
        temporal_layer = TemporalAttention()
        output = temporal_layer(sample_input)
        expected_shape = (batch_size, features)
        
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"   ✅ Temporal Attention works! Shape: {expected_shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Temporal Attention failed: {e}")
    
    # Test 5: Feature Attention
    print("\n📊 Test 5: Feature Attention Layer")
    total_tests += 1
    try:
        feature_layer = FeatureAttention()
        output = feature_layer(sample_input)
        expected_shape = (batch_size, timesteps, features)
        
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"   ✅ Feature Attention works! Shape preserved")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Feature Attention failed: {e}")
    
    # Test 6: Gradient flow test
    print("\n📊 Test 6: Gradient Flow Test")
    total_tests += 1
    try:
        attention_layer = Attention()
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = attention_layer(sample_input)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, sample_input)
        assert gradients is not None, "Gradients are None!"
        assert not tf.reduce_any(tf.math.is_nan(gradients)), "NaN gradients detected!"
        print(f"   ✅ Gradients flow properly! Gradient shape: {gradients.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")
    
    # Test 7: LSTM + Attention Integration
    print("\n📊 Test 7: LSTM + Attention Integration")
    total_tests += 1
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
            Attention(),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        # Test forward pass
        test_input = tf.random.normal((batch_size, timesteps, features))
        prediction = model(test_input)
        
        print(f"   Model summary:")
        model.summary()
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   ✅ LSTM + Attention model works!")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Model integration failed: {e}")
    
    # Test 8: LSTM + Multi-Head Attention integration
    print("\n📊 Test 8: LSTM + Multi-Head Attention Integration")
    total_tests += 1
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Flatten
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
            MultiHeadAttention(num_heads=4),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        test_input = tf.random.normal((batch_size, timesteps, features))
        prediction = model(test_input)
        
        print(f"   Model summary:")
        model.summary()
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   ✅ LSTM + Multi-Head Attention model works!")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Model integration failed: {e}")
    
    # Test 9: LSTM + Temporal Attention Integration
    print("\n📊 Test 9: LSTM + Temporal Attention Integration")
    total_tests += 1
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
            TemporalAttention(),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        test_input = tf.random.normal((batch_size, timesteps, features))
        prediction = model(test_input)
        
        print(f"   Model summary:")
        model.summary()
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   ✅ LSTM + Temporal Attention model works!")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ LSTM + Temporal Attention integration failed: {e}")
    
    # Test 10: Realistic Crop Yield Dimensions
    print("\n📊 Test 10: Realistic Crop Yield Dimensions")
    total_tests += 1
    try:
        # Typical dimensions for crop yield prediction
        model_seq_length = 12  # 12 months of data
        model_features = 50    # Weather + soil + crop features
        
        realistic_input = tf.random.normal((batch_size, model_seq_length, model_features))
        
        # Test with different attention mechanisms
        basic_attn = Attention()
        multihead_attn = MultiHeadAttention(num_heads=5)  # 50/5=10, divisible
        temporal_attn = TemporalAttention()
        feature_attn = FeatureAttention()
        
        basic_out = basic_attn(realistic_input)
        multihead_out = multihead_attn(realistic_input)
        temporal_out = temporal_attn(realistic_input)
        feature_out = feature_attn(realistic_input)
        
        print(f"   Input dimensions: (batch={batch_size}, seq_len={model_seq_length}, features={model_features})")
        print(f"   Basic Attention output: {basic_out.shape}")
        print(f"   Multi-Head Attention output: {multihead_out.shape}")
        print(f"   Temporal Attention output: {temporal_out.shape}")
        print(f"   Feature Attention output: {feature_out.shape}")
        print(f"   ✅ All attention layers work with realistic dimensions!")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Realistic dimensions test failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(f"📊 TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("="*60)
    
    if tests_passed == total_tests:
        print("✅ All attention layers are ready to use in your model!")
        print("\n💡 Usage Tips:")
        print("   1. Basic Attention: Best for extracting temporal patterns")
        print("   2. Multi-Head Attention: Best for complex, multi-pattern relationships")
        print("   3. Temporal Attention: Best for time-series with clear temporal patterns")
        print("   4. Feature Attention: Best for feature selection and interpretation")
        print("\n📝 Recommended configuration for crop yield:")
        print("   model = Sequential([")
        print("       LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),")
        print("       Attention(),  # or MultiHeadAttention(num_heads=4)")
        print("       Dense(64, activation='relu'),")
        print("       Dropout(0.3),")
        print("       Dense(1)")
        print("   ])")
    else:
        print(f"⚠️ {total_tests - tests_passed} tests failed. Please check the errors above.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    # Run tests
    success = test_attention_layers()
    
    if success:
        print("\n🎉 Attention layers are ready for integration with your crop yield prediction model!")
    else:
        print("\n❌ Please fix the issues before using attention layers in your model.")