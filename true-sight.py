import numpy as np
import pandas as pd
import tensorflow as tf
from utils import ClearOutput, get_input_shapes
from preprocessing import Preprocessor
from models import AdditiveDecomposition, TrueSight
from statsforecast.models import AutoARIMA, SeasonalNaive

num_time_steps = 50
seasonal_lenght = 12
forecast_horizon = 12

dataset = []
for i in range(20):
    linear_slope = np.random.normal(0, 1)
    seasonal_amplitude = np.random.randint(1, 10)
    noise_amplitude = np.random.randint(1, 10)
    irregularity_amplitude = np.random.randint(1, 10)
    time = np.arange(num_time_steps)
    linear_trend = linear_slope * time
    seasonal_cycle = seasonal_amplitude * np.sin(2 * np.pi * time / seasonal_lenght)
    noise = np.random.normal(0, noise_amplitude, size=num_time_steps)
    irregularities = np.random.normal(0, irregularity_amplitude, size=num_time_steps)
    dataset.append(linear_trend + seasonal_cycle + noise + irregularities)
dataset = np.array(dataset)
dataset[dataset < 0] = 0

dates = pd.date_range(start='2020-01-01', periods=num_time_steps, freq='MS')
ids = np.arange(dataset.shape[0])
df = []
for i, ts in enumerate(dataset):
    df.append(pd.DataFrame({"unique_id": np.tile(ids[i], len(ts)), "ds": dates, "y": ts}))
df = pd.concat(df)

preprocessor = Preprocessor(df)
X_train, Y_train, ids_train, X_val, Y_val, ids_val, models = preprocessor.make_dataset(
    forecast_horizon = 12, 
    season_length = 12,
    date_freq = "MS", 
    models = [AdditiveDecomposition, AutoARIMA, SeasonalNaive], 
    fallback_model = SeasonalNaive)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True, monitor = "val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 1e-6, patience = 25, verbose = 1),
    ClearOutput()
]

input_shapes = get_input_shapes(X_train)
truesight = TrueSight(models, input_shapes, forecast_horizon = forecast_horizon)
truesight.fit(X_train, Y_train, X_val, Y_val, batch_size = 1, epochs = 100, verbose = True, callbacks = callbacks)