import __future__

import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from statsforecast import StatsForecast

from typing import Any
from truesight.containers import Dataset, TimeSeries
from truesight.utils import get_x_y
from tqdm import tqdm

warnings.filterwarnings("ignore", module="statsforecast.arima")


class Preprocessor():

    def __init__(
        self,
        df: pd.DataFrame,
        date_freq: str,
        date_col: str = "ds",
        target_col: str = "y",
        id_col: str | list = "unique_id",
        weight_col: str | None = None,
        category_cols: list = [],
        descriptors_cols: list = [],
        verbose: bool = True,
        multi_category_warning: bool = True
    ) -> None:

        self.df = df.copy()
        self.date_freq = date_freq
        self.has_categories = False
        self.has_descriptors = False
        self.forecast_horizon = None

        if isinstance(id_col, str):
            self.df.rename(columns={date_col: "ds", id_col: "unique_id", target_col: "y"}, inplace=True)
        elif isinstance(id_col, list):
            self.df.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
            self.df["unique_id"] = self.df[id_col].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        else:
            raise Exception("Invalid id_col. Please use a str or a list of str.")

        if len(category_cols) > 0:

            if isinstance(category_cols, str):
                category_cols = [category_cols]
            elif not isinstance(category_cols, list):
                raise Exception("Invalid category_cols. Please use a str or a list of str.")

            for col in category_cols:
                if col not in self.df.columns:
                    raise Exception(f"Category column {col} not found in dataframe.")

            if tf.config.get_visible_devices('GPU') == []:
                warnings.warn(
                    'No GPU detected. When using categories, the model will use a half-transformer architecture. '
                    'This will be slow on CPU.',
                    UserWarning
                )

            for col in category_cols:
                self.df[col] = self.df[col].astype(str)
            self.df['category'] = self.df[category_cols].apply(
                lambda row: ' '.join(row.values.astype(str)), axis=1
            )
            self.has_categories = True

        if len(descriptors_cols) > 0:

            if isinstance(descriptors_cols, str):
                descriptors_cols = [descriptors_cols]
            elif not isinstance(descriptors_cols, list):
                raise Exception("Invalid descriptors_cols. Please use a str or a list of str.")

            for col in descriptors_cols:
                if col not in self.df.columns:
                    raise Exception(f"Descriptor column {col} not found in dataframe.")

            if tf.config.get_visible_devices('GPU') == []:
                warnings.warn(
                    'No GPU detected. When using descriptors, the model will use a half-transformer architecture. '
                    'This will be slow on CPU.',
                    UserWarning
                )

            for col in descriptors_cols:
                self.df[col] = self.df[col].astype(str)
            self.df['descriptors'] = self.df[descriptors_cols].apply(
                lambda row: ' '.join(row.values.astype(str)), axis=1
            )
            self.has_descriptors = True

        if weight_col is not None:

            if not isinstance(weight_col, str):
                raise Exception("Invalid weight_col. Please use a str.")

            if weight_col not in self.df.columns:
                raise Exception(f"Weight column {weight_col} not found in dataframe.")

            self.df.rename(columns={weight_col: "weight"}, inplace=True)
        else:
            self.df["weight"] = 1

        if self.has_categories and self.has_descriptors:
            self.df = self.df[['ds', 'unique_id', 'category', 'descriptors', 'weight', 'y']]
        elif self.has_categories:
            self.df = self.df[['ds', 'unique_id', 'category', 'weight', 'y']]
        elif self.has_descriptors:
            self.df = self.df[['ds', 'unique_id', 'descriptors', 'weight', 'y']]
        else:
            self.df = self.df[['ds', 'unique_id', 'weight', 'y']]

        self.df = self.make_dataframe(date_freq, verbose=verbose, multi_category_warning=multi_category_warning)
        self.df = self.df.sort_values("ds")

        if self.has_categories:
            self.category_vectorizer = self.get_vectorizer(self.df['category'].unique())

        if self.has_descriptors:
            self.descriptor_vectorizer = self.get_vectorizer(self.df['descriptors'].unique())

    def make_dataframe(
        self,
        date_freq: str,
        verbose: bool = True,
        multi_category_warning: bool = True
    ) -> pd.DataFrame:

        result_df = []
        groups = self.df.groupby("unique_id")
        min_date = self.df['ds'].min()
        max_date = self.df['ds'].max()
        for unique_id, group in tqdm(groups, total=len(groups), disable=not verbose):

            if self.has_categories:
                category = group['category'].unique()
                if len(category) > 1:
                    if multi_category_warning:
                        warnings.warn(
                            f'The unique_id {unique_id} has {len(category)} categories: {category}. '
                            f'Only the first one will be used.',
                            UserWarning
                        )

            date_range = pd.date_range(start=min_date, end=max_date, freq=date_freq)
            group = group.groupby(
                pd.Grouper(key="ds", freq=date_freq)
            ).sum().reindex(date_range).fillna(0).reset_index().rename(columns={"index": "ds"})
            group['unique_id'] = unique_id
            if self.has_categories:
                group['category'] = category[0]             # type: ignore
            if self.has_descriptors:
                group['descriptors'].replace(0, 'null', regex=True, inplace=True)

            group['weight'] = group['weight'].sum()
            result_df.append(group)

        return pd.concat(result_df)

    def get_vectorizer(
        self,
        corpus: np.ndarray
    ) -> tf.keras.layers.TextVectorization:

        seq_lenght = self.get_seq_lenght(corpus)
        vectorizer = tf.keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            output_mode='int',
            output_sequence_length=seq_lenght,
        )
        vectorizer.adapt(corpus)
        return vectorizer

    def get_seq_lenght(
        self,
        corpus: np.ndarray
    ) -> int:

        seq_lenght = 0
        for x in corpus:
            if len(x.split(' ')) > seq_lenght:
                seq_lenght = len(x)
        return seq_lenght

    def make_dataset(
        self,
        forecast_horizon: int,
        spliting_method: str = "forecast_horizon",
        train_split: float | None = None,
        stats_models: list = [],
        ml_models: list = [],
        fallback_model: Any = None,
        verbose: bool = True,
        nan_warnings: bool = True
    ) -> tuple:

        self.forecast_horizon = forecast_horizon

        if spliting_method == "forecast_horizon":
            if train_split is not None:
                warnings.warn(
                    'The train_split parameter is ignored when using forecast_horizon (default) spliting_method.',
                    UserWarning
                )
            dates = self.df['ds'].sort_values().unique()
            training_dates = dates[:-self.forecast_horizon]
            validation_dates = dates[self.forecast_horizon:]
            self.train_df = self.df[self.df.ds.isin(training_dates)]
            self.val_df = self.df[self.df.ds.isin(validation_dates)]
        elif spliting_method == "random_picking":
            if train_split is None:
                raise Exception("train_split parameter is required when using random random_picking.")
            self.train_df = self.df.sample(frac=train_split)
            self.val_df = self.df.drop(self.train_df.index)
        else:
            raise Exception("Invalid spliting_method. Please use 'forecast_horizon' or 'random_picking'.")

        self.train_dataset = self.get_dataset(self.train_df, stats_models, ml_models, fallback_model, verbose=verbose)
        self.val_dataset = self.get_dataset(self.val_df, stats_models, ml_models, fallback_model, verbose=verbose)

        self.train_dataset.assert_dataset(nan_warnings)
        self.val_dataset.assert_dataset(nan_warnings)

        return self.train_dataset, self.val_dataset

    def get_dataset(
        self,
        local_df: pd.DataFrame,
        stats_models: list,
        ml_models: list,
        fallback_model: Any = None,
        verbose: bool = True
    ) -> Dataset:

        dates = local_df['ds'].sort_values().unique()
        train_df = local_df[local_df['ds'].isin(dates[:-self.forecast_horizon])]

        if len(stats_models) > 0:
            print("Generating StatsForecast...")
            sf = StatsForecast(
                df=train_df[['unique_id', 'ds', 'y']],
                models=stats_models,
                freq=self.date_freq,
                n_jobs=-1,
                fallback_model=fallback_model,
            )
            stats_forecasts_df = sf.forecast(h=self.forecast_horizon)
            stats_forecasts_df.fillna(0, inplace=True)

        if len(ml_models) > 0:
            ml_forecasts_df = []
            for ml_model in ml_models:
                model_name = ml_model.__class__.__name__
                print(f"Generating {model_name} forecasts...")
                ids, X_train, y_train = get_x_y(self.forecast_horizon, train_df)
                ml_model.fit(X_train, y_train)
                pred = ml_model.predict(X_train)
                for idx, unique_id in enumerate(ids):
                    ml_forecasts_df.append(
                        pd.DataFrame(
                            {
                                'unique_id': np.repeat(unique_id, self.forecast_horizon),
                                'ds': dates[-self.forecast_horizon:],
                                model_name: np.squeeze(pred[idx])
                            }
                        )
                    )
            ml_forecasts_df = pd.concat(ml_forecasts_df)
            ml_forecasts_df = ml_forecasts_df.groupby(['unique_id', 'ds'], as_index=False).sum()

        if len(stats_models) + len(ml_models) == 0:
            raise Warning("No models were provided.")
        elif len(stats_models) > 0 and len(ml_models) > 0:
            aux_df = pd.merge(ml_forecasts_df, stats_forecasts_df, on=['unique_id', 'ds'], how='left')
        elif len(stats_models) > 0:
            aux_df = stats_forecasts_df
        else:
            aux_df = ml_forecasts_df

        models = aux_df.columns[2:].tolist()
        return_dataset = Dataset(
            forecast_horizon=self.forecast_horizon,
            oberved_data_lenght=len(dates[:-self.forecast_horizon]),
            ground_truth_lenght=len(dates[-self.forecast_horizon:]),
            models=models,
            descriptor_vocab_size=self.descriptor_vectorizer.vocabulary_size() if self.has_descriptors else 0,
            category_vocab_size=self.category_vectorizer.vocabulary_size() if self.has_categories else 0
        )

        groups = local_df.groupby("unique_id")
        for unique_id, group in tqdm(groups, total=len(groups), disable=not verbose):

            oberved_data = tf.convert_to_tensor(
                group[group['ds'].isin(dates[:-self.forecast_horizon])]['y'].to_numpy()         # type: ignore
            )
            ground_truth = tf.convert_to_tensor(
                group[group['ds'].isin(dates[-self.forecast_horizon:])]['y'].to_numpy()         # type: ignore
            )

            weight = group['weight'].mean()

            if self.has_categories:
                category = self.category_vectorizer(group['category'].iloc[0])
            else:
                category = None

            if self.has_descriptors:
                descriptors = self.descriptor_vectorizer(group['descriptors'].iloc[0])
            else:
                descriptors = None

            timeseries = TimeSeries(
                unique_id=str(unique_id),
                datetime_index=dates,
                oberved_data=oberved_data,
                ground_truth=ground_truth,
                weight=weight,
                category_vector=category,
                descriptor_vector=descriptors
            )
            for model in models:
                pred = aux_df[aux_df['unique_id'] == unique_id][model].to_numpy()
                timeseries.add_stats_prediction(model, pred)

            return_dataset.add_timeseries(timeseries)

        return return_dataset
