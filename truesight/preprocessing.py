import __future__
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Any
from truesight.containers import Dataset, TimeSeries
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
            category_cols: list = [],
            descriptors_cols: list = [],
            verbose: bool = True,
            multi_category_warning: bool = True
        ) -> None:

        self.df = df.copy()
        self.date_freq = date_freq
        self.has_categories = False
        self.has_descriptors = False
        
        if isinstance(id_col, str):
            self.df.rename(columns = {date_col: "ds", id_col: "unique_id", target_col: "y"}, inplace = True)
        else:
            self.df.rename(columns = {date_col: "ds", target_col: "y"}, inplace = True)
            self.df["unique_id"] = self.df[id_col].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        if len(category_cols) > 0:
            if tf.config.get_visible_devices('GPU') == []:
                warnings.warn(
                    'No GPU detected. When using categories, the model will use a half-transformer architecture. This will be slow on CPU.',
                    UserWarning
                )
            
            for col in category_cols: self.df[col] = self.df[col].astype(str)
            self.df['category'] = self.df[category_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            self.has_categories = True
        
        if len(descriptors_cols) > 0:
            if tf.config.get_visible_devices('GPU') == []:
                warnings.warn(
                    'No GPU detected. When using descriptors, the model will use a half-transformer architecture. This will be slow on CPU.',
                    UserWarning
                )
            
            for col in descriptors_cols: self.df[col] = self.df[col].astype(str)
            self.df['descriptors'] = self.df[descriptors_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            self.has_descriptors = True
        
        if self.has_categories and self.has_descriptors:
            self.df = self.df[['ds', 'unique_id', 'category', 'descriptors', 'y']]
        elif self.has_categories:
            self.df = self.df[['ds', 'unique_id', 'category', 'y']]
        elif self.has_descriptors:
            self.df = self.df[['ds', 'unique_id', 'descriptors', 'y']]
        else:
            self.df = self.df[['ds', 'unique_id', 'y']]        
        
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
            for unique_id, group in tqdm(groups, total = len(groups), disable = not verbose):
                
                if self.has_categories:
                    category = group['category'].unique()
                    if len(category) > 1:
                        if multi_category_warning: warnings.warn(
                            f'The unique_id {unique_id} has {len(category)} categories: {category}. Only the first one will be used.',
                            UserWarning
                        )
                
                date_range = pd.date_range(start=min_date, end=max_date, freq=date_freq)
                group = group.groupby(pd.Grouper(key = "ds", freq = date_freq)).sum().reindex(date_range).fillna(0).reset_index().rename(columns = {"index": "ds"})
                group['unique_id'] = unique_id
                if self.has_categories:
                    group['category'] = category[0] # type: ignore
                if self.has_descriptors:
                    group['descriptors'].replace(0, 'null', regex=True, inplace=True)

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
            train_split: float = 0.8,
            models: list = [],
            fallback_model = None,
            verbose: bool = True,
            nan_warnings: bool = True
        ) -> tuple:

        self.forecast_horizon = forecast_horizon

        training_ids = np.random.choice(self.df['unique_id'].unique(), int(len(self.df['unique_id'].unique()) * train_split), replace = False)
        self.train_df = self.df[self.df['unique_id'].isin(training_ids)]
        self.val_df = self.df[~self.df['unique_id'].isin(training_ids)]
        
        train_dataset = self.get_dataset(self.train_df, models, fallback_model, verbose = verbose)
        val_dataset = self.get_dataset(self.val_df, models, fallback_model, verbose = verbose)
        
        train_dataset.assert_dataset(nan_warnings)
        val_dataset.assert_dataset(nan_warnings)

        return train_dataset, val_dataset

    def get_dataset(
            self, 
            local_df: pd.DataFrame, 
            models: list[Any] = [], 
            fallback_model: Any = None,
            verbose: bool = True
        ) -> Dataset:
        
        dates = local_df['ds'].sort_values().unique()
        groups = local_df.groupby("unique_id")
        return_dataset = Dataset(
            forecast_horizon = self.forecast_horizon,
            oberved_data_lenght = len(dates[:-self.forecast_horizon]),
            ground_truth_lenght = len(dates[-self.forecast_horizon:]),
            models = [model.__repr__() for model in models], 
            descriptor_vocab_size = self.descriptor_vectorizer.vocabulary_size() if self.has_descriptors else 0, 
            category_vocab_size = self.category_vectorizer.vocabulary_size() if self.has_categories else 0
        )
        for unique_id, group in tqdm(groups, total = len(groups), disable = not verbose):
            
            oberved_data = tf.convert_to_tensor(group[group['ds'].isin(dates[:-self.forecast_horizon])]['y'].to_numpy())
            ground_truth = tf.convert_to_tensor(group[group['ds'].isin(dates[-self.forecast_horizon:])]['y'].to_numpy())

            if self.has_categories:
                category = self.category_vectorizer(group['category'].iloc[0])
            else:
                category = None
            
            if self.has_descriptors:
                descriptors = self.descriptor_vectorizer(group['descriptors'].iloc[0])  
            else:
                descriptors = None    
            
            timeseries = TimeSeries(unique_id, dates, oberved_data, ground_truth, category, descriptors) # type: ignore
            for model in models:
                try:
                    model.fit(group['y'].to_numpy())
                except:
                    if fallback_model is not None:
                        fallback_model.fit(group['y'].to_numpy())
                    else:
                        raise Exception("Model fit failed. Please provide a fallback model.")
                
                timeseries.add_stats_prediction(model.__repr__(), model.predict())
                
            return_dataset.add_timeseries(timeseries)
                
        return return_dataset