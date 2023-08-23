import __future__
import os
import optuna
import json
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from truesight.base import StatisticalForecaster
from truesight.metrics import mse
from truesight.utils import TuneRange, get_x_y
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVR


class AutoTuneML():

    def __init__(
        self,
        model: XGBRegressor | LGBMRegressor | RandomForestRegressor,
        hparams: dict[str, TuneRange],
        n_trials: int = 100,
        n_jobs: int = -1,
        forecast_horizon: int = 1,
    ) -> None:

        self._model = model
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.forecast_horizon = forecast_horizon
        self.hparams = hparams

    def get_hparams(
        self,
        tune_df,
        hparams_path: str | None = None,
    ) -> dict[str, int | float]:

        if hparams_path is not None:
            try:
                with open(hparams_path, 'r', encoding='utf-8') as f:
                    stat_info = os.stat(hparams_path)
                    modification_time = datetime.fromtimestamp(stat_info.st_mtime)
                    print(f'Found hparams file at {hparams_path}, last time modified {modification_time}.')
                    print('To prevent this behaviour, set hparams_path to None.')
                    self.hparams = json.load(f)
                return self.hparams
            except FileNotFoundError:
                print(f'No hparams file found at {hparams_path}. Running auto-tune...')
                pass

        dates = tune_df['ds'].unique()
        training_dates = dates[:-self.forecast_horizon]
        validation_dates = dates[self.forecast_horizon:]
        train_df = tune_df[tune_df['ds'].isin(training_dates)]
        valid_df = tune_df[tune_df['ds'].isin(validation_dates)]
        _, self.X_train, self.y_train = get_x_y(self.forecast_horizon, train_df)
        _, self.X_valid, self.y_valid = get_x_y(self.forecast_horizon, valid_df)
        self.hparams = self.autotune()
        self.save_hparams(hparams_path)

        return self.hparams

    def save_hparams(
        self,
        hparams_path: str
    ) -> None:

        if self.hparams is None:
            raise ValueError('No hyperparameters to save. Run get_hparams first.')

        with open(hparams_path, 'w', encoding='utf-8') as f:
            json.dump(self.hparams, f, ensure_ascii=False, indent=4)

    def autotune(
        self,
    ) -> dict[str, int | float]:

        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)

        return study.best_trial.params

    def objective(
        self,
        trial: optuna.Trial
    ) -> float:

        params = {}
        for key, param in self.hparams.items():
            if param.dtype == float:
                params[key] = trial.suggest_float(key, param.min, param.max)
            elif param.dtype == int:
                params[key] = trial.suggest_int(key, param.min, param.max)
            else:
                raise ValueError(f'Unknown parameter type {param.dtype}')

        self.model = self._model(verbose=0, **params)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_valid)
        score = mse(self.y_valid, y_pred, return_mean=True)

        return score


class LGBMRegressorMultiHorizon(LGBMRegressor):

    def __init__(
        self,
        **kwargs
    ) -> None:

        self.lgbm = LGBMRegressor(**kwargs)
        self.n_windows = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:

        self.n_windows = y.shape[1]

        X_rolled = []
        y_rolled = []
        for i in range(self.n_windows):
            X_rolled.extend(np.hstack((X[:, i:], y[:, :i])))
            y_rolled.extend(y[:, i])

        self.lgbm.fit(X_rolled, y_rolled)

    def predict(
        self,
        X
    ) -> np.ndarray:

        y_pred = np.zeros((X.shape[0], self.n_windows))
        for i in range(self.n_windows):
            if i == 0:
                y_pred[:, i] = self.lgbm.predict(X)
            else:
                y_pred[:, i] = self.lgbm.predict(np.hstack((X[:, i:], y_pred[:, :i])))

        return y_pred


class ApproximateSVR(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        n_components: int = 100,
        C: float = 1.0,
        epsilon: float = 0.1
    ) -> None:

        self.n_components = n_components
        self.C = C
        self.epsilon = epsilon
        self.nystroem = Nystroem(n_components=self.n_components)
        self.svr = LinearSVR(C=self.C, epsilon=self.epsilon)
        self.n_windows = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:

        self.n_windows = y.shape[1]
        X = self.nystroem.fit_transform(X)

        X_rolled = []
        y_rolled = []
        for i in range(self.n_windows):
            X_rolled.extend(np.hstack((X[:, i:], y[:, :i])))
            y_rolled.extend(y[:, i])

        self.svr.fit(X_rolled, y_rolled)

    def predict(
        self,
        X
    ) -> np.ndarray:

        X = self.nystroem.transform(X)
        y_pred = np.zeros((X.shape[0], self.n_windows))
        for i in range(self.n_windows):
            if i == 0:
                y_pred[:, i] = self.svr.predict(X)
            else:
                y_pred[:, i] = self.svr.predict(np.hstack((X[:, i:], y_pred[:, :i])))

        return y_pred


class AdditiveDecomposition(StatisticalForecaster):

    def __init__(
        self,
        season_length: int,
        alias: str = 'AdditiveDecomposition'
    ) -> None:
        
        self.season_length = season_length
        self.trend_coeff: np.ndarray | None = None
        self.seasonality: np.ndarray | None = None
        self.noise: np.ndarray | None = None
        self.alias = alias

    def fit(
        self,
        y: np.ndarray,
        trend_degrees: list = [1, 2]
    ) -> None:

        self.train_lenght = len(y)
        x = np.arange(self.train_lenght)

        best_trend_mse = float('inf')

        for degree in trend_degrees:
            coefficients = np.polyfit(x, y, degree)
            trend = np.polyval(coefficients, x)
            trend_mse = mse(y, trend)

            if trend_mse < best_trend_mse:
                self.trend_coeff = coefficients
                best_trend_mse = trend_mse

        decomposition = seasonal_decompose(y, model='additive', period=self.season_length)
        self.seasonality = decomposition.seasonal
        self.noise = decomposition.resid

    def predict(self, forecast_horizon):

        x = np.arange(self.train_lenght, self.train_lenght + forecast_horizon)

        trend_forecast = np.polyval(self.trend_coeff, x) # type: ignore
        seasonality_forecast = np.tile(self.seasonality[:forecast_horizon], forecast_horizon // self.season_length + 1)[:forecast_horizon] # type: ignore
        noise_forecast = np.random.choice(self.noise, size=forecast_horizon) # type: ignore

        forecast = trend_forecast + seasonality_forecast + noise_forecast

        return forecast

    def __repr__(self) -> str:
        return f'{self.alias}'