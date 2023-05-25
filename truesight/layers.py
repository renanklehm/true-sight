import tensorflow as tf
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int
    ) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

    def compute_mask(
        self, 
        *args, 
        **kwargs
    ) -> tf.Tensor:
        
        return self.embedding.compute_mask(*args, **kwargs)

    def call(
        self, 
        x: tf.Tensor
    ) -> tf.Tensor:
        
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
  
    def positional_encoding(
        self, 
        length: int, 
        depth: int
    ) -> tf.Tensor:

        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

        return tf.cast(pos_encoding, dtype=tf.float32)


class CoherenceLayer(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        context_size: int, 
        timesteps: int,
        **kwargs
    ) -> None:
        
        super(CoherenceLayer, self).__init__(**kwargs)
        self.timesteps = timesteps
        self.context_size = context_size

    def build(
        self, 
        input_shapes: list
    ) -> None:
        self.n = len(input_shapes)
        self.context_list = [tf.keras.layers.Dense(self.context_size) for _ in range(self.n)]
        for i, layer in enumerate(self.context_list):
            layer.build(input_shapes[i])
            
        self.output_layer = tf.keras.layers.Dense(self.timesteps)
        self.output_layer.build((None, self.context_size))
        super(CoherenceLayer, self).build(input_shapes)

    def call(
        self, 
        inputs: list
    ) -> tf.Tensor:
        context_list = [layer(inputs[i]) for i, layer in enumerate(self.context_list)]
        context = tf.stack(context_list, axis=-1)
        context = tf.reduce_max(context, axis=-1)
        output = self.output_layer(context)
        return output


class BaseAttention(tf.keras.layers.Layer):
  
    def __init__(
        self, 
        **kwargs
    ) -> None:

        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  
    def call(
        self, 
        x: tf.Tensor, 
        context: tf.Tensor
    ) -> tf.Tensor:

        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )

        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class GlobalSelfAttention(BaseAttention):

    def call(
        self, 
        x: tf.Tensor
    ) -> tf.Tensor:

        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
  
    def call(
        self, 
        x: tf.Tensor
    ) -> tf.Tensor:

        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        d_model: int, 
        dff: int, 
        dropout_rate: float = 0.1
    ) -> None:
        
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(
        self, 
        input: tf.Tensor,
        training: bool = True
    ) -> tf.Tensor:
        
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.add([input, x])
        x = self.layer_norm(x) 
        return x

class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(
        self,
        *, 
        d_model: int,
        num_heads: int, 
        dff: int, 
        dropout_rate:float = 0.1
    ) -> None:
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(
        self, 
        x: tf.Tensor
    ) -> tf.Tensor:
        
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate:float = 0.1
    ) -> None:
        
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(
        self, 
        x: tf.Tensor, 
        context: tf.Tensor
    ) -> tf.Tensor:
        
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x

class LSTM(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        units: int, 
        dropout_rate:float = 0.1
    ) -> None:
        
        super(LSTM, self).__init__()
        self.units = units
        self.dropout_rate = dropout_rate
        self.normalizer = tf.keras.layers.LayerNormalization()
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(
        self, 
        inputs: tf.Tensor, 
        initial_state:bool = None, 
        training:bool = True
    ) -> tf.Tensor:
        
        normalized_inputs = self.normalizer(inputs)
        if initial_state is None: initial_state = self.lstm_cell.get_initial_state(normalized_inputs)
        lstm_outputs, final_state = tf.keras.layers.RNN(self.lstm_cell)(normalized_inputs, initial_state=initial_state, training=training)
        lstm_outputs = self.dropout(lstm_outputs, training=training)
        outputs = self.add([normalized_inputs, lstm_outputs])
        outputs = self.layer_norm(outputs)
        return outputs
    
class BranchOutput(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        output_size: int, 
        dropout_rate:float = 0.1
    ) -> None:
        
        super(BranchOutput, self).__init__()
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        #self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(
        self, 
        inputs: tf.Tensor,
        training:bool = True
    ) -> tf.Tensor:
        
        inputs = self.flatten(inputs)
        outputs = self.dense(inputs)
        outputs = self.dropout(outputs, training=training)
        #outputs = self.add([inputs, outputs])
        outputs = self.layer_norm(outputs)
        return outputs

class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(
        self,
        *, 
        d_model: int,
        num_heads: int, 
        dff: int, 
        dropout_rate:float = 0.1
    ) -> None:
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(
        self, 
        x: tf.Tensor
    ) -> tf.Tensor:
        
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate:float = 0.1
    ) -> None:
        
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(
        self, 
        x: tf.Tensor, 
        context: tf.Tensor
    ) -> tf.Tensor:
        
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x