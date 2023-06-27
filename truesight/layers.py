import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        filter_size: int,
        context_size: int,
        hidden_size: int, 
        dropout_rate: float = 0.1,
        **kwargs
    ) -> None:
        
        super(FeedForward, self).__init__(**kwargs)
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
        self.reshape_dense = tf.keras.layers.Dense(hidden_size, activation='selu')

    def build(
        self, 
        input_shapes: tuple
    ) -> None:
        
        self.dense.build(input_shapes)
        super(FeedForward, self).build(input_shapes)

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
        
        return x
    
class WeightedSumLayer(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        n_models: int, 
        **kwargs
    ) -> None:
        
        super(WeightedSumLayer, self).__init__(**kwargs)
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
        return output