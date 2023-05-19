import __future__

import os
import json
import optuna
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

class TrueSight:

    def __init__(
            self,
            models: list,
            input_shape: list, 
            forecast_horizon: int,
        ) -> None:
        self.model_folder = 'best_model'
        self.hparams_folder = './hparams'
        os.makedirs(self.hparams_folder, exist_ok=True)
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.models = models
        self.set_hparams()
        self.model = self.get_model(self.hparams)

    def get_model(
            self, 
            hparams: dict = {}
        ) -> tf.keras.Model:
        x_inputs = []
        x_outputs = []
        x = []
        for i in range(len(self.models)):
            x_inputs.append(tf.keras.layers.Input((self.input_shape[i], 1), name = f"input_{i}"))
            x.append(x_inputs[i])
            x[i] = tf.keras.layers.LayerNormalization(epsilon=1e-8)(x[i])
            x[i] = tf.keras.layers.Dense(hparams['hidden_size'], activation='selu', name=f"dense_input_{i}")(x[i])
            x[i] = tf.keras.layers.Dropout(hparams['dropout_rate'], name=f"dropout_{i}")(x[i], training = True)
            x[i] = tf.keras.layers.Conv1D(hparams['num_filters'], kernel_size=hparams['kernel_size'], activation='selu', padding='same', name=f"conv1d_{i}_1")(x[i])
            x[i] = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same', name=f"max_pool_{i}_1")(x[i])
            x[i] = tf.keras.layers.Conv1D(hparams['num_filters'], kernel_size=hparams['kernel_size'], activation='selu', padding='same', name=f"conv1d_{i}_2")(x[i])
            x[i] = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same', name=f"max_pool_{i}_2")(x[i])
            x[i] = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hparams['lstm_units'], return_sequences=True), name=f"bidirectional_lstm_{i}")(x[i])
            x[i] = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hparams['hidden_size'], activation='selu'), name=f"timedistributed_dense_{i}")(x[i])
            x[i] = tf.keras.layers.MultiHeadAttention(hparams['num_heads'], hparams['key_dim'], name = f"multihead_self_attention_{i}")(x[i], x[i])
            x[i] = tf.keras.layers.Flatten(name=f"flatten_{i}")(x[i])
            x_outputs.append(x[i])
        y = tf.keras.layers.Concatenate(name="concatenate")(x_outputs)
        y = tf.keras.layers.Dense(hparams['hidden_size'], activation='selu', name="dense_output")(y)
        y = tf.keras.layers.Dropout(hparams['dropout_rate'], name="dropout_output")(y, training = True)
        y = tf.keras.layers.Dense(self.forecast_horizon, activation='selu', name="output_dense")(y)
        model = tf.keras.Model(x_inputs, y, name="TrueSight")
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def fit(
            self, 
            X_train: list, 
            Y_train: np.ndarray, 
            X_val: list, 
            Y_val: np.ndarray, 
            batch_size: int = 128, 
            epochs: int = 100, 
            callbacks: list = [],
            save_best_model: bool = True,
            verbose: bool = True
        ):
        self.history = self.model.fit(
            x = X_train,
            y = Y_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = [X_val, Y_val],
            callbacks = callbacks,
            verbose = verbose
            )
        if save_best_model: self.model.save(self.model_folder)

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
            yhat.append(self.model.predict(X, batch_size=batch_size, verbose=verbose))
        yhat = np.array(yhat)

        if return_quantiles: yhat = np.quantile(yhat, np.linspace(0, 1, n_quantiles), axis=0)
        else: yhat = np.mean(yhat, axis=0)
        return yhat

    def load_model(self):
        if not os.path.exists(self.model_folder): raise Exception("No model found")
        self.model = tf.keras.models.load_model(self.model_folder)
    
    def load_hparams(self):
        if not os.path.exists(f'{self.hparams_folder}/best_hparams.json'): raise Exception("No best hparams found, please run the auto_tune() first")
        with open(f'{self.hparams_folder}/best_hparams.json', 'r') as file: self.hparams = json.load(file)

    def set_hparams(
            self,
            num_filters: int = 64,
            kernel_size: int = 3,
            lstm_units: int = 64,
            hidden_size: int = 64,
            num_heads: int = 4,
            key_dim: int = 64,
            learning_rate: float = 0.001,
            dropout_rate: float = 0.2,
        ) -> None:
        self.hparams = {
            'num_filters': num_filters,
            'kernel_size': kernel_size,
            'lstm_units': lstm_units,
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'key_dim': key_dim,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
        }

    def objective(self, trial):
        num_filters = trial.suggest_int("num_filters", self.min_num_filters, self.max_num_filters)
        kernel_size = trial.suggest_int("kernel_size", self.min_kernel_size, self.max_kernel_size)
        lstm_units = trial.suggest_int("lstm_units", self.min_lstm_units, self.max_lstm_units)
        hidden_size = trial.suggest_int("hidden_size", self.min_hidden_size, self.max_hidden_size)
        num_heads = trial.suggest_int("num_heads", self.min_num_heads, self.max_num_heads)
        key_dim = trial.suggest_int("key_dim", self.min_key_dim, self.max_key_dim)
        dropout_rate = trial.suggest_float("dropout_rate", self.min_dropout_rate, self.max_dropout_rate )
        learning_rate = trial.suggest_float("learning_rate", self.min_learning_rate, self.max_learning_rate)
        hparams = {
            'num_filters': num_filters,
            'kernel_size': kernel_size,
            'lstm_units': lstm_units,
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'key_dim': key_dim,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
        }

        model = self.get_model(hparams)
        model.fit(self.X_train, self.Y_train, epochs = self.epochs, batch_size = self.batch_size, verbose = 0)
        score = model.evaluate(self.X_val, self.Y_val, batch_size = self.batch_size, verbose = 0)
        score = np.array(score).mean()
        with open(f'{self.hparams_folder}/{datetime.now().strftime("%Y%m%d%H%M%S")}-{score}.json', 'w') as file: json.dump(hparams, file)
        return score

    def auto_tune(
            self,
            X_train: list,
            Y_train: np.ndarray,
            X_val: list,
            Y_val: np.ndarray,
            n_trials: int,
            batch_size: int = 128,
            epochs: int = 5,
            min_num_filters: int = 32,
            max_num_filters: int = 256,
            min_kernel_size: int = 3,
            max_kernel_size: int = 21,
            min_lstm_units: int = 32,
            max_lstm_units: int = 256,
            min_hidden_size: int = 128,
            max_hidden_size: int = 1024,
            min_num_heads: int = 2,
            max_num_heads: int = 10,
            min_key_dim: int = 16,
            max_key_dim: int = 128,
            min_learning_rate: float = 0.0001,
            max_learning_rate: float = 0.01,
            min_dropout_rate: float = 0.05,
            max_dropout_rate: float = 0.4,
        ) -> dict:

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_num_filters = min_num_filters
        self.max_num_filters = max_num_filters
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_lstm_units = min_lstm_units
        self.max_lstm_units = max_lstm_units
        self.min_hidden_size = min_hidden_size
        self.max_hidden_size = max_hidden_size
        self.min_num_heads = min_num_heads
        self.max_num_heads = max_num_heads
        self.min_key_dim = min_key_dim
        self.max_key_dim = max_key_dim
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_dropout_rate = min_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        
        file_list = os.listdir(self.hparams_folder)
        for file_name in file_list:
            file_path = os.path.join(self.hparams_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        self.hparams = study.best_trial.params
        with open(f'{self.hparams_folder}/best_hparams.json', 'w') as file: json.dump(self.hparams, file)
        return self.hparams

    def plot_history(self):
        if not hasattr(self, 'history'): raise Exception('No history found. Please train the model first.')
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()