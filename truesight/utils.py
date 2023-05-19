import numpy as np
import pandas as pd

def get_input_shapes(X):
    input_shapes = []
    for x in X:
        input_shapes.append(x.shape[1])
    return input_shapes

def generate_syntetic_data(
        num_time_steps: int,
        seasonal_lenght: int,
        num_series: int,
    ) -> pd.DataFrame:
    dataset = []
    for _ in range(num_series):
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
        dataset.append(scale_factor * (linear_trend + seasonal_cycle + noise + irregularities))
    dataset = np.array(dataset)
    dataset[dataset < 0] = 0
    dates = pd.date_range(start='2020-01-01', periods=num_time_steps, freq='MS')
    ids = np.arange(dataset.shape[0])
    df = []
    for i, ts in enumerate(dataset):
        df.append(pd.DataFrame({"unique_id": np.tile(ids[i], len(ts)), "ds": dates, "y": ts}))
    return pd.concat(df)