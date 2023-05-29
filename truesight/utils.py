import os
import json
import inspect
import optuna
import random
import string
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator
from statsforecast.models import _TS
from datetime import datetime
from truesight.core import TrueSight
from truesight.base import StatisticalForecaster
from truesight.containers import Dataset

class ModelWrapper:
    def __init__(
        self, 
        model: Any, 
        horizon: int | None = None,
        alias: str | None = None,
        **kwargs
    ) -> None:
        
        valid_args = inspect.signature(model.__init__).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        self.model = model(**filtered_kwargs)
        self.forecast_horizon = horizon
        if alias is None:
            self.alias = self.model.__repr__()
        else:
            self.alias = alias
    
    def fit(
        self, 
        X: np.ndarray
    ) -> None:
        
        self.X = X
        if isinstance(self.model, BaseEstimator):
            if len(X) % 2 != 0:
                X = np.insert(X, -1, X.mean())
            mid_size = int(len(X) / 2)
            self.model.fit(np.expand_dims(X[:-mid_size], -1), np.expand_dims(X[mid_size:], -1)) # type: ignore
        elif isinstance(self.model, _TS):
            self.model.fit(np.squeeze(X))  # type: ignore
        elif isinstance(self.model, StatisticalForecaster):
            self.model.fit(X)
        else:
            try:
                self.model.fit(X)
            except:
                raise ValueError("Unsupported model type.")
    
    def predict(
        self
    ) -> tf.Tensor:
        
        if isinstance(self.model, BaseEstimator):
            return_prediction = np.squeeze(self.model.predict(np.expand_dims(self.X[-self.forecast_horizon:], -1))) # type: ignore
        elif isinstance(self.model, _TS):
            return_prediction = np.squeeze(self.model.predict(h=self.forecast_horizon)['mean'])  # type: ignore
        elif isinstance(self.model, StatisticalForecaster):
            return_prediction = np.squeeze(self.model.predict(self.forecast_horizon))  # type: ignore
        else:
            try:
                return_prediction = np.squeeze(self.model.predict(self.forecast_horizon))
            except:
                raise ValueError("Unsupported model type.")
        
        return tf.convert_to_tensor(return_prediction)
            
    def __repr__(
        self
    ) -> str:
        
        return self.alias

class AutoTune:
    
    def __init__(
        self,
        model: tf.keras.Model | None = None,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
        min_lr: float = 1e-5,
        max_lr: float = 1e-2,
        min_dropout: float = 0.0,
        max_dropout: float = 0.5,
        min_filter_size: int = 32,
        max_filter_size: int = 256,
        min_contex_size: int = 128,
        max_contex_size: int = 512,
        min_hidden_size: int = 512,
        max_hidden_size: int = 1024
    ) -> None:
            
        self.model = TrueSight if model is None else model
        self.optimizer = tf.keras.optimizers.Adam if optimizer is None else optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.min_filter_size = min_filter_size
        self.max_filter_size = max_filter_size
        self.min_contex_size = min_contex_size
        self.max_contex_size = max_contex_size
        self.min_hidden_size = min_hidden_size
        self.max_hidden_size = max_hidden_size
        
    def tune(
        self,
        dataset: Dataset,
        n_trials: int = 10,
        epochs: int = 10,
        batch_size: int = 32,
        save_trials_hparams: bool = False,
        hparams_folder: str = 'hparams'
    ) -> tuple[dict[str, Any], float]:
        
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size        
        self.save_trials_hparams = save_trials_hparams
        self.hparams_folder = hparams_folder
        
        os.makedirs(self.hparams_folder, exist_ok=True)
        file_list = os.listdir(self.hparams_folder)
        for file_name in file_list:
            file_path = os.path.join(self.hparams_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        self.hparams = study.best_trial.params
        with open(f'{self.hparams_folder}/best_hparams.json', 'w') as file: json.dump(self.hparams, file)
        
        return self.hparams, self.optimizer(self.hparams['lr'])
        
    def objective(
        self, 
        trial: optuna.Trial
    ) -> float:
        
        lr = trial.suggest_float('lr', self.min_lr, self.max_lr)
        dropout_rate = trial.suggest_float('dropout_rate', self.min_dropout, self.max_dropout)
        filter_size = trial.suggest_int('filter_size', self.min_filter_size, self.max_filter_size)
        context_size = trial.suggest_int('context_size', self.min_contex_size, self.max_contex_size)
        hidden_size = trial.suggest_int('hidden_size', self.min_hidden_size, self.max_hidden_size)
        
        model: TrueSight = self.model(
            dataset = self.dataset,
            filter_size = filter_size, 
            context_size = context_size, 
            hidden_size = hidden_size, 
            dropout_rate = dropout_rate
        )
        opt = self.optimizer(learning_rate = lr)
        model.compile(optimizer = opt, loss = 'mse') # type: ignore
        model.fit(train_dataset = self.dataset, epochs = self.epochs, batch_size = self.batch_size, verbose = 0) # type: ignore
        score = model.evaluate(dataset = self.dataset, batch_size = self.batch_size, verbose = 0) # type: ignore
        score = np.array(score).mean()
        
        hparams = {
            'lr': lr,
            'dropout_rate': dropout_rate,
            'filter_size': filter_size,
            'context_size': context_size,
            'hidden_size': hidden_size,
            'score': score
        }
        
        if self.save_trials_hparams:
            with open(f'{self.hparams_folder}/{datetime.now().strftime("%Y%m%d%H%M%S")}-{score}.json', 'w') as file: json.dump(hparams, file)
        return score

def generate_syntetic_data(
    num_time_steps: int,
    seasonal_lenght: int,
    num_series: int,
    start_date: str = '2020-01-01',
    freq: str = 'MS'
) -> pd.DataFrame:
    
    dataset = []
    for _ in range(num_series):
        starting_point = np.random.randint(0, 20)
        linear_slope = np.random.normal(0, 1)
        seasonal_amplitude = np.random.randint(1, 10)
        noise_amplitude = np.random.randint(1, 10)
        irregularity_amplitude = np.random.randint(1, 10)
        scale_factor = np.random.randint(1, 10)
        time = np.arange(num_time_steps)
        linear_trend = linear_slope * time
        seasonal_cycle = seasonal_amplitude * np.sin(2 * np.pi * time / seasonal_lenght)
        noise = np.random.normal(0, noise_amplitude, size=num_time_steps)
        irregularities = np.random.normal(0, irregularity_amplitude, size=num_time_steps)
        dataset.append(scale_factor * (linear_trend + seasonal_cycle + noise + irregularities) + starting_point)
    dataset = np.array(dataset)
    dataset[dataset < 0] = 0
    dates = pd.date_range(start=start_date, periods=num_time_steps, freq=freq)
    ids = [get_random_string(10) for _ in range(num_series)]
    df = []
    for i, ts in enumerate(dataset):
        df.append(pd.DataFrame({"unique_id": np.tile(ids[i], len(ts)), "ds": dates, "y": ts}))
    return pd.concat(df)

def generate_time_series(
    length: int, 
    seasonality_cycles: list[int], 
    seasonality_amplitudes: list[int], 
    trend_slopes: list[float], 
    trend_change_indexes: list[int]
) -> np.ndarray:
    
    series = np.zeros(length)
    
    trend = np.arange(length) * trend_slopes[0]
    for i in range(1, len(trend_slopes)):
        start_index = trend_change_indexes[i-1]
        end_index = trend_change_indexes[i]
        trend[start_index:end_index] = np.arange(end_index - start_index) * trend_slopes[i] + trend[start_index-1]
    
    num_seasons = len(seasonality_cycles)
    for i in range(num_seasons):
        cycle = seasonality_cycles[i]
        amplitude = seasonality_amplitudes[i]
        seasonality = amplitude * np.sin(2 * np.pi * np.arange(length) / cycle)
        series += seasonality
    
    series += trend
    return series

def get_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))