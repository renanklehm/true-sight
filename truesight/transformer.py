import tensorflow as tf
from truesight.layers import PositionalEmbedding, EncoderLayer, DecoderLayer


class Transformer(tf.keras.Model):
    def __init__(
        self, 
        *,
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        dff: int,
        input_vocab_size: int, 
        target_vocab_size: int, 
        dropout_rate:float = 0.1
    ) -> None:
        
        super().__init__()
        self._d_model = d_model
        self.encoder = Encoder(
            num_layers=num_layers, 
            d_model=d_model,
            num_heads=num_heads, 
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate
        )
        self.decoder = Decoder(
            num_layers=num_layers, 
            d_model=d_model,
            num_heads=num_heads, 
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.final_layer.build((None, d_model))

    def call(
        self, 
        inputs: tf.Tensor,
    ) -> tf.Tensor:

        context, x = inputs
        x = tf.tile(x, [1, 1, self._d_model])
        context = self.encoder(context)
        x = self.decoder(x, context)
        output = self.final_layer(x)
        output = tf.reduce_mean(output, axis=1)
        output = tf.expand_dims(output, axis=-1)
        return output

class Encoder(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        *, 
        num_layers: int, 
        d_model: int, 
        num_heads: int,
        dff: int, 
        vocab_size: int, 
        dropout_rate:float = 0.1
    ) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)       
        self.enc_layers = []
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate)
            )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self, 
        x: tf.Tensor
    ) -> tf.Tensor:
        
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x

class Decoder(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        *, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        dff: int, 
        vocab_size: int,
        dropout_rate:float = 0.1
    ) -> None:
        
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(dropout_rate)        
        self.dec_layers = []
        for _ in range(num_layers):
            self.dec_layers.append(DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate)
            )
        self.last_attn_scores = None

    def call(
        self, 
        x: tf.Tensor, 
        context: tf.Tensor
    ) -> tf.Tensor:

        x = self.dropout(x)
        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x