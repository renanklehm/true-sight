import __future__

import os
import json
import optuna
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from truesight.layers import FeedForward, WeightedSumLayer

class TrueSight(tf.keras.Model):

    def __init__(
            self,
            models: list,
            forecast_horizon: int,
            filter_size: int = 64,
            context_size: int = 256,
            hidden_size: int = 512,
            dropout_rate: float = 0.1,
        ) -> None:
        
        super(TrueSight, self).__init__()
        self.models = models
        self.n_models = len(models)
        self.forecast_horizon = forecast_horizon
        self.filter_size = filter_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        self.branches = {}
        for i in range(self.n_models):
            self.branches[models[i]] = tf.keras.layers.Dense(context_size, activation='selu', name=f'branch_{models[i]}')
        self.weighted_sum = WeightedSumLayer(n_models=self.n_models, name='weighted_sum')
        self.ff = FeedForward(filter_size=filter_size, context_size=context_size, hidden_size=hidden_size, dropout_rate=dropout_rate, name='feed_forward')
        self.output_layer = tf.keras.layers.Dense(forecast_horizon, name='output')
    
    def set_hparams(
        self,
        hparams: dict,
    ) -> None:
        
        self.__init__(self.models, self.forecast_horizon, hparams['filter_size'], hparams['context_size'], hparams['hidden_size'], hparams['dropout_rate'])
    
    def build(
        self, 
        input_shape: list,
    ) -> None:
        
        for idx, branch in enumerate(self.branches.values()):
            branch.build(input_shape[idx])
        self.ff.build((None, 1))
        self.output_layer.build((None, self.hidden_size))
        return 
    
    def call(
        self, 
        inputs: list,
    ) -> tf.Tensor:
        
        outputs = []
        for idx, model in enumerate(self.models):
            outputs.append(self.branches[model](inputs[idx]))
        outputs = self.weighted_sum(outputs)
        outputs = self.ff(outputs, training=True)
        outputs = self.output_layer(outputs)
        outputs = tf.clip_by_value(outputs, clip_value_min=0.0, clip_value_max=tf.float32.max)
        return outputs # type: ignore
    
    def fit(
        self,
        **kwargs
    ) -> None:
        
        self.history = super(TrueSight, self).fit(**kwargs)
    
    def plot_training_history(
        self,
    ) -> None:
        
        if not hasattr(self, 'history'): raise Exception('No history found. Please train the model first.')
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        plt.show(
    )

    def predict(
        self,
        X: list,
        n_repeats: int = 1,
        batch_size: int = 128,
        n_quantiles: int = 10,
        return_quantiles: bool = False,
        verbose: bool = True
    ) -> np.ndarray:

        yhat = []
        for i in range(n_repeats):
            yhat.append(super(TrueSight, self).predict(X, batch_size=batch_size, verbose=verbose))
        yhat = np.array(yhat)

        if return_quantiles or n_quantiles > 1: 
            yhat = np.quantile(yhat, np.linspace(0, 1, n_quantiles), axis=0)
        else: 
            yhat = np.mean(yhat, axis=0)
        
        return yhat
