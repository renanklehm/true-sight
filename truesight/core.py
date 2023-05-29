import __future__

import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from truesight.layers import NumericalFeedForward, ContextLayer, PositionalEmbedding, EncoderLayer
from truesight.containers import Dataset


class TrueSight(tf.keras.Model):

    def __init__(
            self,
            dataset: Dataset,
            filter_size: int = 64,
            n_encoder_layers: int = 2,
            n_encoder_heads: int = 8,
            context_size: int = 256,
            hidden_size: int = 512,
            dropout_rate: float = 0.1,
        ) -> None:
        
        super(TrueSight, self).__init__()
        warnings.warn("Due to compatibility issues and performance improvements, the next version of TrueSight (v0.0.5a) will change its entire framework to PyTorch", DeprecationWarning)
        self._dataset_model = dataset
        self.models = dataset.models
        self.forecast_horizon = dataset.forecast_horizon
        self.category_vocab_size = dataset.category_vocab_size
        self.descriptor_vocab_size = dataset.descriptor_vocab_size
        self.has_categories = dataset.has_categories
        self.has_descriptors = dataset.has_descriptors
        
        self.filter_size = filter_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        self.branches = {}
        if dataset.has_categories:
            self.category_encoder = Encoder(
                num_layers=n_encoder_layers, 
                d_model=context_size, 
                num_heads=n_encoder_heads, 
                dff=hidden_size, 
                vocab_size=dataset.category_vocab_size, 
                dropout_rate=dropout_rate, name='category_encoder')
            self.category_flatten = tf.keras.layers.Flatten(name='category_flatten')
            self.branches['categories'] = tf.keras.layers.Dense(context_size, activation='selu', name='categories')
        
        if dataset.has_descriptors:
            self.descriptor_encoder = Encoder(
                num_layers=n_encoder_layers, 
                d_model=context_size, 
                num_heads=n_encoder_heads, 
                dff=hidden_size, 
                vocab_size=dataset.descriptor_vocab_size, 
                dropout_rate=dropout_rate, name='descriptor_encoder')
            self.descriptor_flatten = tf.keras.layers.Flatten(name='descriptor_flatten')
            self.branches['descriptors'] = tf.keras.layers.Dense(context_size, activation='selu', name='descriptors')
        
        for i in range(len(self.models)):
            self.branches[self.models[i]] = tf.keras.layers.Dense(context_size, activation='selu', name=f'{self.models[i]}')
            
        self.branches['observed_data'] = tf.keras.layers.Dense(context_size, activation='selu', name='observed_data')

        self.weighted_sum = ContextLayer(n_models=len(self.branches), name='weighted_sum')
        self.ff = NumericalFeedForward(filter_size=filter_size, context_size=context_size, hidden_size=hidden_size, dropout_rate=dropout_rate, name='feed_forward')
        self.output_layer = tf.keras.layers.Dense(self.forecast_horizon, activation='relu', name='output')
    
    def set_hparams(
        self,
        hparams: dict,
    ) -> None:
        
        self.__init__(self._dataset_model, hparams['filter_size'], hparams['context_size'], hparams['hidden_size'], hparams['dropout_rate'])
    
    def call(
        self, 
        inputs: list,
    ) -> tf.Tensor:
        
        outputs = []
        for idx, branch in enumerate(self.branches.values()):
            temp = inputs[idx]
            if self.has_categories and idx == 0:
                temp = self.category_encoder(temp)
                temp = self.category_flatten(temp)
            elif self.has_descriptors and idx == 1:
                temp = self.descriptor_encoder(temp)
                temp = self.descriptor_flatten(temp)                
            outputs.append(branch(temp))
                
        outputs = self.weighted_sum(outputs)
        outputs = self.ff(outputs, training=True)
        outputs = self.output_layer(outputs)
        return outputs # type: ignore
    
    def fit(
        self,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        **kwargs
    ) -> None:
        
        if train_dataset is None:
            if 'x' not in kwargs.keys() or 'y' not in kwargs.keys():
                raise Exception("You need to provide either a dataset or x and y.")
        else:
            if val_dataset is None:
                self.history = super(TrueSight, self).fit(
                    x = train_dataset.get_x(), y = train_dataset.get_y(),
                    **kwargs)
            else:
                self.history = super(TrueSight, self).fit(
                    x = train_dataset.get_x(), y = train_dataset.get_y(),
                    validation_data = [val_dataset.get_x(), val_dataset.get_y()],
                    **kwargs)

    def predict(
        self,
        X: tf.Tensor | np.ndarray | None = None,
        dataset: Dataset | None = None,
        n_repeats: int = 1,
        batch_size: int = 128,
        n_quantiles: int = 10,
        return_quantiles: bool = False,
        verbose: bool = True
<<<<<<< Updated upstream
    ) -> np.ndarray:
=======
    ) -> np.ndarray | Dataset:
>>>>>>> Stashed changes

        if X is None and dataset is None:
            raise Exception("Please provide either X or a dataset.")
        if dataset is not None:
            if X is not None:
                warnings.warn("Both X and dataset were provided. X will be ignored.", UserWarning)
<<<<<<< Updated upstream
            _dataset = dataset.copy()
            _X = _dataset.get_x()
        if X is not None:
=======
            if not isinstance(dataset, Dataset):
                raise Exception("dataset must be of type Dataset.")
            _dataset = dataset.copy()
            _X = _dataset.get_x()
        if X is not None:
            if not isinstance(X, (tf.Tensor, np.ndarray)):
                raise Exception("X must be either a tf.Tensor or np.ndarray.")
>>>>>>> Stashed changes
            _X = X
            
        yhat = []
        for _ in range(n_repeats):
            yhat.append(super(TrueSight, self).predict(x = _X, batch_size=batch_size, verbose=verbose)) # type: ignore
        yhat = np.array(yhat)

        if return_quantiles or n_quantiles > 1: 
            yhat = np.quantile(yhat, np.linspace(0, 1, n_quantiles), axis=0)
        else: 
            yhat = np.mean(yhat, axis=0)
        
        if dataset is not None:
<<<<<<< Updated upstream
            dataset.add_predictions(yhat, has_quartiles=return_quantiles)  
=======
            dataset.add_predictions(yhat, has_quartiles=return_quantiles)
            return dataset
>>>>>>> Stashed changes
        
        return yhat
    
    def evaluate(
        self,
<<<<<<< Updated upstream
        dataset: Dataset,
        **kwargs
    ) -> float:
        
        return super(TrueSight, self).evaluate(
            x = dataset.get_x(), y = dataset.get_y(),
            **kwargs) # type: ignore
=======
        dataset: Dataset | None = None,
        **kwargs
    ) -> float:
        
        if dataset is None:
            if 'x' not in kwargs.keys() or 'y' not in kwargs.keys():
                raise Exception("You need to provide either a dataset or x and y.")
            return super(TrueSight, self).evaluate(**kwargs) # type: ignore
        else:
            if 'x' in kwargs.keys() or 'y' in kwargs.keys():
                warnings.warn("Both dataset and x and y were provided. Dataset will be used.", UserWarning)
            return super(TrueSight, self).evaluate(x = dataset.get_x(), y = dataset.get_y(), **kwargs) # type: ignore
>>>>>>> Stashed changes
    
    def plot_training_history(
        self,
    ) -> None:
        
        if not hasattr(self, 'history'): raise Exception('No history found. Please train the model first.')
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()
    
class Encoder(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int,
        dff: int, 
        vocab_size: int, 
        dropout_rate:float = 0.1,
        **kwargs
    ) -> None:
        
        super(Encoder, self).__init__(**kwargs)
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
        
        x = self.pos_embedding(x) # type: ignore
        x = self.dropout(x) # type: ignore
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x