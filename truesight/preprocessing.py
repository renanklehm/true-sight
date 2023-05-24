import __future__
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from truesight.utils import TimeIt
from tqdm import tqdm

warnings.filterwarnings("ignore", module="statsforecast.arima")

class Preprocessor():

    def __init__(
            self, 
            df: pd.DataFrame
        ) -> None:
        self.df = df.copy()
        self.df = self.df.sort_values("ds")
        self.df = self.df.reset_index(drop = True)
    
    def make_dataset(
            self,
            forecast_horizon: int, 
            season_length: int,
            date_freq: str = "MS",
            train_split: float = 0.8,
            models: list = [],
            fallback_model = None,
            verbose: bool = True,
            seq_lenght: int = None,
        ) -> tuple:

        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.date_freq = date_freq

        training_ids = np.random.choice(self.df.unique_id.unique(), int(len(self.df.unique_id.unique()) * train_split), replace = False)
        self.train_df = self.df[self.df.unique_id.isin(training_ids)]
        self.val_df = self.df[~self.df.unique_id.isin(training_ids)]

        dates = self.df['ds'].sort_values().unique()

        if verbose: t = TimeIt("Fitting training data")
        train_forecast_df = self.get_statistical_forecast(self.train_df[self.train_df['ds'].isin(dates[:-self.forecast_horizon])], models, fallback_model, verbose = verbose)
        if verbose: t.get_time()

        if verbose: t = TimeIt("Fitting validation data")
        val_forecast_df = self.get_statistical_forecast(self.val_df[self.val_df['ds'].isin(dates[:-self.forecast_horizon])], models, fallback_model, verbose = verbose)
        if verbose: t.get_time()
        
        self.train_df = pd.merge(self.train_df, train_forecast_df, on = ["unique_id", "ds"], how = "left")
        self.val_df = pd.merge(self.val_df, val_forecast_df, on = ["unique_id", "ds"], how = "left")

        self.X_train, self.Y_train, self.models = self.format_dataset(self.train_df)
        self.X_val, self.Y_val, _ = self.format_dataset(self.val_df)

        self.vectorizer = self.get_vectorizer(self.df, seq_lenght)

        if verbose: t = TimeIt("Vectorizing training data")
        self.X_train[-1] = self.vectorizer(self.X_train[-1]).numpy()
        self.X_train.append(np.roll(self.Y_train, -1, axis = -2))
        if verbose: t.get_time()

        if verbose: t = TimeIt("Vectorizing validation data")
        self.X_val[-1] = self.vectorizer(self.X_val[-1]).numpy()
        self.X_val.append(np.roll(self.Y_val, -1, axis = -2))
        if verbose: t.get_time()

        self.input_shape = self.get_input_shapes(self.X_train)

    def get_input_shapes(self, X):
        input_shapes = []
        for x in X:
            input_shapes.append(x.shape[1])
        return input_shapes

    def get_vectorizer(
            self, 
            local_df: pd.DataFrame,
            seq_lenght: int = None,
        ) -> tf.keras.layers.TextVectorization:

        corpus = local_df['unique_id'].unique()
        if seq_lenght is None:
            seq_lenght = 0
            for x in corpus:
                if len(x.split(' ')) > seq_lenght:
                    seq_lenght = len(x)
        vectorizer = tf.keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            output_mode='int',
            output_sequence_length=seq_lenght,
        )
        vectorizer.adapt(corpus)
        return vectorizer

    def format_dataset(
            self, 
            local_df: pd.DataFrame
        ) -> tuple:
        pivot = pd.pivot_table(local_df, index = "unique_id", columns = "ds")
        models = np.unique(pivot.columns.get_level_values(0))
        x = []
        for model in models:
            if (model == "y"):
                x.append(np.expand_dims(pivot[model].iloc[:,:-self.forecast_horizon].to_numpy(), -1))
            else:
                x.append(np.expand_dims(pivot[model].to_numpy(), -1))
        y = np.expand_dims(pivot["y"].iloc[:,-self.forecast_horizon:].to_numpy(), -1)
        x.append(pivot.index)
        return x, y, models

    def get_statistical_forecast(
            self, 
            local_df: pd.DataFrame, 
            models: list = [], 
            fallback_model = None,
            verbose: bool = True
        ) -> pd.DataFrame:
        groups = local_df.groupby("unique_id")
        return_df = []
        for unique_id, group in tqdm(groups, total = len(groups), disable = not verbose):
            output_dates = pd.date_range(group.ds.max(), periods = self.forecast_horizon + 1, freq = self.date_freq)[1:]
            output_ids = [unique_id] * len(output_dates)
            for model in models:
                try:
                    model.fit(group['y'].to_numpy())
                except:
                    if fallback_model is not None:
                        model = fallback_model(season_length = self.season_length)
                        model.fit(group['y'].to_numpy())
                    else:
                        raise Exception("Model fit failed. Please provide a fallback model.")
                return_df.append(pd.DataFrame({
                    "unique_id": output_ids,
                    "ds": output_dates,
                    model.__repr__(): model.predict(),
                }))
        return_df = pd.concat(return_df)
        return_df = return_df.groupby(['unique_id', 'ds'], as_index = False).sum()
        return return_df