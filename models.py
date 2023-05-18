import __future__
import optuna
import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from metrics import mse
from scipy.optimize import curve_fit

class AdditiveDecomposition():
    def __init__(
            self, 
            season_length: int, 
            smooth_factor: float = 0.6
        ) -> None:
        self.season_length = season_length
        self.smooth_factor = smooth_factor
        self.alias = "Additive Decomposition"
    
    def __repr__(
            self
        ) -> str:
        return self.alias

    def fit(
            self, 
            y: np.ndarray, 
            trend_estimators: list = ["linear", "log"]
        ) -> None:

        self.trend_estimators = []
        for trend_estimator in trend_estimators:
            match trend_estimator:
                case "linear":
                    self.trend_estimators.append(self.linear_eq)
                case "log":
                    self.trend_estimators.append(self.log_eq)
                case "power":
                    self.trend_estimators.append(self.power_eq)
                case "exp":
                    self.trend_estimators.append(self.exp_eq)
                case _:
                    raise ValueError("Trend estimator not found")

        self.y = y
        self.input_size = len(y)
        self.total_size = self.input_size * 2
        self.trend_line = self.get_trend()
        self.season_line = self.get_seasonality()
    
    def predict(
            self, 
            fcst_horizon: int
        ) -> dict:
        if (fcst_horizon < 1): raise ValueError("The forecast horizon must be greater than 0")
        if (fcst_horizon > self.input_size): raise ValueError("The forecast horizon cannot be greater than the input size")
        prediction = self.trend_line + self.season_line
        prediction[prediction < 0] = 0
        prediction = prediction[-fcst_horizon:]
        prediction_return = {}
        prediction_return['mean'] = prediction
        return prediction_return

    def get_trend(
            self
        ) -> np.ndarray:
        x = np.arange(1, self.total_size + 1)
        temp_trends = []
        for trend_estimator in self.trend_estimators:
            try:
                popt, _ = curve_fit(trend_estimator, x[:self.input_size], self.y)
                temp_trends.append(trend_estimator(x, *popt))
            except:
                pass
        best_score = np.inf
        best_trend = None
        for trend in temp_trends:
            score = mse(self.y, trend[:self.input_size], return_mean=True)
            if (score < best_score):
                best_trend = trend
                best_score = score
        trend_line = best_trend
        trend_line[trend_line < 0] = 0
        return trend_line

    def get_seasonality(
            self
        ) -> np.ndarray:
        seasonality = []
        dataset_season = self.y - self.trend_line[:self.input_size]
        cycles = int(self.input_size/self.season_length)
        for i in range(self.season_length):
            step_val = []
            for step in range(cycles):
                step_val.append(dataset_season[i + (step * self.season_length)])
            step_val = np.array(step_val).sum(axis = 0)
            step_val = np.nan_to_num(step_val / cycles)
            seasonality.append(step_val)
        seasonality = np.asarray(seasonality)
        seasonality = np.tile(seasonality, self.total_size)[:self.total_size]
        if (self.smooth_factor > 0): seasonality = self.smooth(seasonality, self.smooth_factor)
        return seasonality

    def plot_test(
            self, 
            fcst_horizon: int
        ) -> None:
        pred = self.prediction(fcst_horizon)
        _, ax = plt.subplots(figsize=(10, 6), dpi=100)
        range_total = np.arange(0, self.total_size)
        range_input = np.arange(0, self.input_size)
        range_output = np.arange(self.input_size, self.total_size)
        ax.plot(range_total, self.trend_line, label = "trend")
        ax.plot(range_input, self.y, label = "history")
        ax.plot(range_output, pred, label = "prediction")
        ax.plot(range_total, self.trend_line + self.season_line, label = "prediction")
        ax.plot(self.season_line, label = "seasonality")
        ax.legend()
        plt.show()

    def smooth(
            self, 
            scalars: np.ndarray, 
            weight: float
        ) -> list: 
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point 
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def linear_eq(
            self, 
            x: np.ndarray, 
            a: float, 
            c: float
        ) -> np.ndarray:
        return a*x + c

    def log_eq(
            self, 
            x: np.ndarray, 
            a: float, 
            c: float
        ) -> np.ndarray:
        return a*np.log(x) + c

    def power_eq(
            self, 
            x: np.ndarray, 
            a: float,
            b: float,
            c: float
        ) -> np.ndarray:
        return a*np.power(x, b) + c
    
    def exp_eq(
            self, 
            x: np.ndarray, 
            a: float,
            b: float,
            c: float
        ) -> np.ndarray:
        return np.power(a, b * x) + c

class TrueSight:

    def __init__(
            self,
            models: list,
            input_shape: list, 
            forecast_horizon: int,
            folder_path: str = "best_model"
        ) -> None:
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.folder_path = folder_path
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
            x[i] = tf.keras.layers.LayerNormalization()(x[i])
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
        y = tf.keras.layers.Masking()(y)
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
        if save_best_model: self.model.save(self.folder_path)

    def load_model(self):
        if not os.path.exists(self.folder_path): raise Exception("No model found")
        self.model = tf.keras.models.load_model(self.folder_path)

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

        model = self.get_model(self.X_train, self.forecast_horizon, hparams)
        model.fit(self.X_train, self.Y_train, epochs = self.epochs, batch_size = self.batch_size)
        score = model.evaluate(self.X_val, self.Y_val, verbose=1, batch_size = self.batch_size)
        with open(f'/hparams/{datetime.now().strftime("%Y%m%d-%H%M%S")}-score:{score}.json', 'w') as file: json.dump(hparams, file)
        return score

    def find_hparams(
            self,
            X_train: list,
            Y_train: np.ndarray,
            X_val: list,
            Y_val: np.ndarray,
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
        
        os.makedirs('/hparams', exist_ok=True)
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=100)
        with open(f'/hparams/best_hparams.json', 'w') as file: json.dump(study.best_trial.params.items(), file)
        return study.best_trial.params.items()

    def plot_history(self):
        if not hasattr(self, 'history'): raise Exception('No history found. Please train the model first.')
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()