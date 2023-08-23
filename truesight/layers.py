import tensorflow as tf
import numpy as np


class NumericalFeedForward(tf.keras.layers.Layer):

    def __init__(
        self,
        filter_size: int,
        context_size: int,
        hidden_size: int,
        dropout_rate: float = 0.1,
        **kwargs
    ) -> None:

        super(NumericalFeedForward, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(context_size, activation='selu')
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=7, activation='selu')
        self.mp1 = tf.keras.layers.MaxPool1D(pool_size=5)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=5, activation='selu')
        self.mp2 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.conv3 = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=3, activation='selu')
        self.mp3 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.flatten = tf.keras.layers.Flatten()
        self.reshape_dense = tf.keras.layers.Dense(hidden_size, activation='relu')

    def call(
        self,
        input: tf.Tensor,
        training: bool = True
    ) -> tf.Tensor:

        x = tf.expand_dims(input, axis=-1)
        x = self.dense(x)
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.dropout2(x, training=training)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.dropout3(x, training=training)
        x = self.lstm(x)
        x = self.flatten(x)
        x = self.reshape_dense(x)

        return x  # type: ignore


class ContextLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        n_models: int,
        **kwargs
    ) -> None:

        super(ContextLayer, self).__init__(**kwargs)
        self.num_tensors = n_models
        self.w = self.add_weight(
            shape=(n_models,),
            initializer='ones',
            trainable=True,
        )

    def call(
        self,
        inputs: list
    ) -> tf.Tensor:

        weighted_inputs = []
        for i in range(self.num_tensors):
            weighted_inputs.append(inputs[i] * self.w[i])
        output = tf.add_n(weighted_inputs)

        return output  # type: ignore


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        **kwargs
    ) -> None:

        super(PositionalEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

    def compute_mask(
        self,
        *args,
        **kwargs
    ) -> tf.Tensor:

        return self.embedding.compute_mask(*args, **kwargs)     # type: ignore

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:

        length = tf.shape(x)[1]                                 # type: ignore
        x = self.embedding(x)                                   # type: ignore
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))    # type: ignore
        x = x + self.pos_encoding[tf.newaxis, :length, :]       # type: ignore

        return x                                                # type: ignore

    def positional_encoding(
        self,
        length: int,
        depth: int | float
    ) -> tf.Tensor:

        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)          # type: ignore


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
        )                                                       # type: ignore

        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])                          # type: ignore
        x = self.layernorm(x)                                   # type: ignore
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

        x = self.add([x, attn_output])                          # type: ignore
        x = self.layernorm(x)                                   # type: ignore
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
            use_causal_mask=True
        )
        x = self.add([x, attn_output])                          # type: ignore
        x = self.layernorm(x)                                   # type: ignore
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
        return x                                                # type: ignore


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ) -> None:

        super(EncoderLayer, self).__init__(**kwargs)
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

        x = self.self_attention(x)                              # type: ignore
        x = self.ffn(x)                                         # type: ignore
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ) -> None:

        super(DecoderLayer, self).__init__(**kwargs)
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

        x = self.causal_self_attention(x=x)                 # type: ignore
        x = self.cross_attention(x=x, context=context)      # type: ignore
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)                                     # type: ignore
        return x
