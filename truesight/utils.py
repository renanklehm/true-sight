import os
import json
import optuna
import random
import string
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Any
from datetime import datetime
from truesight.core import TrueSight
from truesight.containers import Dataset


class TuneRange:

    def __init__(
        self,
        min: float | int,
        max: float | int,
        step: int | None = 1
    ) -> None:

        if not (
                (isinstance(min, int) and isinstance(max, int)) or
                (isinstance(min, float) and isinstance(max, float))
        ):
            raise ValueError('min and max must have the same datatype, either int or float.')

        self.min = min
        self.max = max
        self.dtype = type(min)


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
        with open(f'{self.hparams_folder}/best_hparams.json', 'w') as file:
            json.dump(self.hparams, file)

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
            dataset=self.dataset,
            filter_size=filter_size,
            context_size=context_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
        opt = self.optimizer(learning_rate=lr)
        model.compile(optimizer=opt, loss='mse')                                                            # type: ignore
        model.fit(train_dataset=self.dataset, epochs=self.epochs, batch_size=self.batch_size, verbose=0)    # type: ignore
        score = model.evaluate(dataset=self.dataset, batch_size=self.batch_size, verbose=0)                 # type: ignore
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
            with open(f'{self.hparams_folder}/{datetime.now().strftime("%Y%m%d%H%M%S")}-{score}.json', 'w') as file:
                json.dump(hparams, file)
        return score


def get_x_y(
    forecast_horizon: int,
    tune_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:

    groups = tune_df.groupby('unique_id')
    X = []
    y = []
    indexes = []
    for idx, group in groups:
        indexes.append(idx)
        X.append(group[:-forecast_horizon]['y'].values)
        y.append(group[-forecast_horizon:]['y'].values)

    return np.array(indexes), np.array(X), np.array(y)


def generate_syntetic_data(
    num_time_steps: int,
    num_series: int,
    start_date: str,
    freq: str,
    start_intercept: float,
    seasonality_cycles: list[int],
    seasonality_amplitudes: list[float],
    trend_slopes: list[float],
    trend_indexes: list[int],
    noise_std: float,
    noise_amplitude: float,
    irregularity_std: float,
    allow_negatives: bool = False
) -> pd.DataFrame:

    if trend_indexes[0] != 0:
        if len(trend_indexes) == len(trend_slopes) - 1:
            trend_indexes.insert(0, 0)
        else:
            raise ValueError('trend_indexes must start with 0 or have the first index ommited')
    else:
        if len(trend_indexes) != len(trend_slopes):
            raise ValueError('trend_indexes and trend_slopes must have the same length')

    dataset = pd.DataFrame(index=pd.date_range(start=start_date, periods=num_time_steps, freq=freq))
    for _ in range(num_series):
        series = np.zeros(num_time_steps)
        seasonality = np.zeros(num_time_steps)

        _seasonality_amplitudes = []
        for i in range(len(seasonality_amplitudes)):
            _seasonality_amplitudes.append(seasonality_amplitudes[i] + np.random.normal(0, irregularity_std))

        _trend_slopes = []
        for i in range(len(trend_slopes)):
            _trend_slopes.append(trend_slopes[i] + np.random.normal(0, irregularity_std / 2))

        _trend_indexes = []
        for i in range(len(trend_indexes)):
            if i == 0:
                _trend_indexes.append(0)
            else:
                _trend_indexes.append(trend_indexes[i] + int(np.random.normal(0, irregularity_std)))
                _trend_indexes[i] = np.clip(_trend_indexes[i], 0, num_time_steps - 1)

        trend = []
        intercept = start_intercept
        for i, slope in enumerate(_trend_slopes):
            start_index = _trend_indexes[i]
            end_index = _trend_indexes[i+1] if i+1 < len(_trend_indexes) else len(series)
            x = np.arange(end_index - start_index)
            y = slope * x + intercept
            intercept = y[-1]
            trend.extend(y)
        trend = np.array(trend)

        for idx, season in enumerate(seasonality_cycles):
            seasonality += _seasonality_amplitudes[idx] * np.sin(2 * np.pi * np.arange(num_time_steps) / season)

        series += trend
        series += seasonality
        series += np.random.normal(0, noise_std, size=num_time_steps) * noise_amplitude

        if not allow_negatives:
            series = np.clip(series, 0, None)

        dataset[get_random_string(10)] = series

    dataset = dataset.copy()
    dataset.reset_index(inplace=True, names='ds')
    dataset = dataset.melt(id_vars=['ds'], var_name='unique_id', value_name='y')
    return dataset


def get_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
