import __future__

import numpy as np
from truesight.base import StatisticalForecaster
from truesight.metrics import mse
from statsmodels.tsa.seasonal import seasonal_decompose

class AdditiveDecomposition(StatisticalForecaster):

    def __init__(
        self, 
        season_length: int,
        alias: str = 'AdditiveDecomposition'
    ) -> None:
        
        self.season_length = season_length
        self.trend_coeff = None
        self.seasonality = None
        self.noise = None
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

        trend_forecast = np.polyval(self.trend_coeff, x)
        seasonality_forecast = np.tile(self.seasonality[:forecast_horizon], forecast_horizon // self.season_length + 1)[:forecast_horizon]
        noise_forecast = np.random.choice(self.noise, size=forecast_horizon)

        forecast = trend_forecast + seasonality_forecast + noise_forecast

        return forecast

    def __repr__(self) -> str:
        return f'{self.alias}'