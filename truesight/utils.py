import inspect
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator
from truesight.base import StatisticalForecaster
from statsforecast.models import _TS

class ModelWrapper:
    def __init__(
        self, 
        model: object, 
        horizon: int = None, 
        **kwargs
    ) -> None:
        
        valid_args = inspect.signature(model.__init__).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        self.model = model(**filtered_kwargs)
        self.forecast_horizon = horizon
    
    def fit(
        self, 
        X: np.ndarray
    ) -> None:
        
        self.X = X
        if isinstance(self.model, BaseEstimator):
            if len(X) % 2 != 0:
                X = np.insert(X, -1, X.mean())
            mid_size = int(len(X) / 2)
            self.model.fit(np.expand_dims(X[:-mid_size], -1), np.expand_dims(X[mid_size:], -1))
        elif isinstance(self.model, _TS):
            self.model.fit(np.squeeze(X))
        elif isinstance(self.model, StatisticalForecaster):
            self.model.fit(X)
        else:
            try:
                self.model.fit(X)
            except:
                raise ValueError("Unsupported model type.")
    
    def predict(
        self
    ) -> np.ndarray:
        
        if isinstance(self.model, BaseEstimator):
            return np.squeeze(self.model.predict(np.expand_dims(self.X[-self.forecast_horizon:], -1)))
        elif isinstance(self.model, _TS):
            return np.squeeze(self.model.predict(h=self.forecast_horizon)['mean'])
        elif isinstance(self.model, StatisticalForecaster):
            return np.squeeze(self.model.predict(self.forecast_horizon))
        else:
            try:
                np.squeeze(self.model.predict(self.forecast_horizon))
            except:
                raise ValueError("Unsupported model type.")
            
    def __repr__(
        self
    ) -> str:
        
        return self.model.__repr__()

def generate_syntetic_data(
        num_time_steps: int,
        seasonal_lenght: int,
        num_series: int,
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
    dates = pd.date_range(start='2020-01-01', periods=num_time_steps, freq='MS')
    ids = np.arange(dataset.shape[0])
    df = []
    for i, ts in enumerate(dataset):
        df.append(pd.DataFrame({"unique_id": np.tile(ids[i], len(ts)), "ds": dates, "y": ts}))
    return pd.concat(df)

class TimeIt:
    def __init__(self, label, print_start = True):
        self.now = datetime.now()
        self.label = label
        self.print_start = print_start
        if self.print_start:
            print(f"{label}...", end = ' ')
    
    def get_time(self):
        if self.print_start:
            print(f"Done in {(datetime.now() - self.now).total_seconds()} s")
        else:
            print(f"{self.label}...Done in {(datetime.now() - self.now).total_seconds()} s")