import __future__

import os
import json
import optuna
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
import matplotlib.pyplot as plt
from datetime import datetime
from truesight.preprocessing import Preprocessor
from truesight.layers import FeedForward, LSTM, BranchOutput
from truesight.transformer import Transformer

class TrueSight:

    def __init__(
            self,
            preprocessor: Preprocessor,
            model_folder: str = 'best_model',
            hparams_folder: str = './hparams',
        ) -> None:
        self.model_folder = model_folder
        self.hparams_folder = hparams_folder
        os.makedirs(self.hparams_folder, exist_ok=True)

        self.preprocessor = preprocessor
        self.set_hparams()
        self.model = self.get_model(self.hparams)

    def get_model(
            self, 
            hparams: dict = {}
        ) -> tf.keras.Model:
        x_inputs = []
        x_outputs = []
        x = []
        for i in range(len(self.preprocessor.models)):
            x_inputs.append(tf.keras.layers.Input((self.preprocessor.input_shape[i], 1), name = f"input_{i}"))
            x.append(x_inputs[i])
            x[i] = FeedForward(hparams['dff'], hparams['dropout_rate'])(x[i])
            x[i] = tf.keras.layers.LSTM(hparams['d_model'], return_sequences=True)(x[i])
            x[i] = BranchOutput(hparams['dff'])(x[i])
            x_outputs.append(x[i])
            
        x_inputs.append(tf.keras.layers.Input((self.preprocessor.input_shape[-2],), name = "transformer_input"))
        x_inputs.append(tf.keras.layers.Input((self.preprocessor.input_shape[-1],), name = "shifted_ground_truth"))
        
        x.append(x_inputs[-2])
        x[-2] = tf.keras.layers.Embedding(self.preprocessor.vectorizer.vocabulary_size(), hparams['d_model'] * hparams['num_heads'])(x[-2])
        x[-2] = Transformer(
            hparams['num_layers'], 
            hparams['d_model'], 
            hparams['num_heads'], 
            hparams['dff'], 
            self.preprocessor.vectorizer.vocabulary_size(), 
            hparams['dropout_rate'])((x[-2], x[-1]))
        x[-2] = BranchOutput(hparams['dff'])(x[-2])
        x_outputs.append(x[-2])

        y = tf.keras.layers.Concatenate(name="concatenate")(x_outputs)
        y = FeedForward(
            self.preprocessor.forecast_horizon,
            hparams['dropout_rate'], 
            name="feedforward")(y)
        model = tf.keras.Model(x_inputs, y, name="TrueSight")
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def fit(
            self,
            batch_size: int = 128, 
            epochs: int = 100, 
            callbacks: list = [],
            save_best_model: bool = True,
            verbose: bool = True
        ):

        self.history = self.model.fit(
            x = self.preprocessor.X_train,
            y = self.preprocessor.Y_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = [self.preprocessor.X_val, self.preprocessor.Y_val],
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
            num_layers: int = 4,
            lstm_units: int = 128,
            d_model: int = 128,
            dff: int = 512,
            num_heads: int = 8,
            dropout_rate: int = 0.1,
            learning_rate: float = 0.001,
        ) -> None:
        
        self.hparams = {
            'num_layers': num_layers,
            'lstm_units': lstm_units,
            'd_model': d_model,
            'dff': dff,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }

    def objective(self, trial):
        num_layers = trial.suggest_int("num_layers", self._min_num_layers, self._max_num_layers)
        lstm_units = trial.suggest_int("lstm_units", self._min_lstm_units, self._max_lstm_units)
        d_model = trial.suggest_int("d_model", self._min_d_model, self._max_d_model)
        dff = trial.suggest_int("dff", self._min_dff, self._max_dff)
        num_heads = trial.suggest_int("num_heads", self._min_num_heads, self._max_num_heads)
        dropout_rate = trial.suggest_float("dropout_rate", self._min_dropout_rate, self._max_dropout_rate)
        learning_rate = trial.suggest_float("learning_rate", self._min_learning_rate, self._max_learning_rate)
        hparams = {
            'num_layers': num_layers,
            'lstm_units': lstm_units,
            'd_model': d_model,
            'dff': dff,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }
        model = self.get_model(hparams)
        model.fit(self.preprocessor.X_train, self.preprocessor.Y_train, epochs = self.epochs, batch_size = self.batch_size, verbose = 0)
        score = model.evaluate(self.preprocessor.X_val, self.preprocessor.Y_val, batch_size = self.batch_size, verbose = 0)
        score = np.array(score).mean()
        with open(f'{self.hparams_folder}/{datetime.now().strftime("%Y%m%d%H%M%S")}-{score}.json', 'w') as file: json.dump(hparams, file)
        return score

    def auto_tune(
            self,
            n_trials: int,
            batch_size: int = 128,
            epochs: int = 5,
            min_num_layers: int = 1,
            max_num_layers: int = 10,
            min_lstm_units: int = 32,
            max_lstm_units: int = 512,
            min_d_model: int = 32,
            max_d_model: int = 512,
            min_dff: int = 32,
            max_dff: int = 512,
            min_num_heads: int = 1,
            max_num_heads: int = 10,
            min_dropout_rate: float = 0.1,
            max_dropout_rate: float = 0.5,
            min_learning_rate: float = 0.0001,
            max_learning_rate: float = 0.01,
        ) -> dict:

        self._batch_size = batch_size
        self._epochs = epochs
        self._min_num_layers = min_num_layers
        self._max_num_layers = max_num_layers
        self._min_lstm_units = min_lstm_units
        self._max_lstm_units = max_lstm_units
        self._min_d_model = min_d_model
        self._max_d_model = max_d_model
        self._min_dff = min_dff
        self._max_dff = max_dff
        self._min_num_heads = min_num_heads
        self._max_num_heads = max_num_heads
        self._min_dropout_rate = min_dropout_rate
        self._max_dropout_rate = max_dropout_rate
        self._min_learning_rate = min_learning_rate
        self._max_learning_rate = max_learning_rate
        
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