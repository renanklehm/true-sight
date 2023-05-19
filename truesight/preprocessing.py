import __future__
import warnings
import pandas as pd
import numpy as np
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
            verbose: bool = True
        ) -> tuple:

        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.date_freq = date_freq

        training_ids = np.random.choice(self.df.unique_id.unique(), int(len(self.df.unique_id.unique()) * train_split), replace = False)
        self.train_df = self.df[self.df.unique_id.isin(training_ids)]
        self.val_df = self.df[~self.df.unique_id.isin(training_ids)]
        dates = self.df['ds'].sort_values().unique()
        train_forecast_df = self.get_statistical_forecast(self.train_df[self.train_df['ds'].isin(dates[:-self.forecast_horizon])], models, fallback_model, verbose = verbose)
        val_forecast_df = self.get_statistical_forecast(self.val_df[self.val_df['ds'].isin(dates[:-self.forecast_horizon])], models, fallback_model, verbose = verbose)
        self.train_df = pd.merge(self.train_df, train_forecast_df, on = ["unique_id", "ds"], how = "left")
        self.val_df = pd.merge(self.val_df, val_forecast_df, on = ["unique_id", "ds"], how = "left")
        X_train, Y_train, ids_train, models = self.format_dataset(self.train_df)
        X_val, Y_val, ids_val, _ = self.format_dataset(self.val_df)

        return X_train, Y_train, ids_train, X_val, Y_val, ids_val, models

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
        ids = pivot.index
        return x, y, ids, models

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
                    model = model(season_length = self.season_length)
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
                    model.__repr__(): model.predict(self.forecast_horizon)['mean'],
                }))
        return_df = pd.concat(return_df)
        return_df = return_df.groupby(['unique_id', 'ds'], as_index = False).sum()
        return return_df