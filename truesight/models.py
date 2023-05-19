import __future__

import numpy as np
import matplotlib.pyplot as plt
from truesight.metrics import mse
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