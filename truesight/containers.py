from __future__ import annotations
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
<<<<<<< Updated upstream
=======
import pickle
>>>>>>> Stashed changes
import matplotlib.pyplot as plt
from typing import Any


class Dataset():
    
    def __init__(
        self,
        forecast_horizon: int,
        oberved_data_lenght: int,
        ground_truth_lenght: int,
        models: list[str],
        descriptor_vocab_size: int = 0,
        category_vocab_size: int = 0
    ) -> None:
        
        self.oberved_data_lenght = oberved_data_lenght
        self.ground_truth_lenght = ground_truth_lenght
        self.models = models
        self.forecast_horizon = forecast_horizon
        self.timeseries = []
        self.has_predictions = False
        self.is_asserted = False
        
        if category_vocab_size > 0:
            self.has_categories = True
            self.category_vocab_size = category_vocab_size
        
        if descriptor_vocab_size > 0:
            self.has_descriptors = True
            self.descriptor_vocab_size = descriptor_vocab_size
    
    def add_timeseries(
        self,
        timeseries: TimeSeries
    ) -> None:
        
        if isinstance(timeseries, TimeSeries):
            if timeseries.unique_id in [ts.unique_id for ts in self.timeseries]:
                raise Exception("Time series already in dataset.")
            if timeseries.oberved_data_lenght < self.oberved_data_lenght:
                raise Exception("Time series oberved data is shorter than the dataset definition.")
            if timeseries.ground_truth_lenght < self.forecast_horizon:
                raise Exception("Time series ground truth is shorter than the dataset definition.")
            if self.has_categories and timeseries.category_vector is None:
                raise Exception("Time series has no category vector, this dataset expects a category vector.")
            if self.has_descriptors and timeseries.descriptor_vector is None:
                raise Exception("Time series has no descriptor vector, this dataset expects a descriptor vector.")
            for model in self.models:
                if model not in timeseries.stats_predictions.keys():
                    raise Exception(f"Missing model {model} for time series {timeseries.unique_id}.")
                if timeseries.stats_predictions[model].shape[0] != self.forecast_horizon:
                    raise Exception("Time series has a model with a different forecast horizon than the dataset definition.")
            self.timeseries.append(timeseries)
        else:
            raise Exception("Parameter received is not a TimeSeries object.")
    
    def assert_dataset(
        self,
        nan_warnings: bool = True
    ) -> None:
        
        if len(self.timeseries) == 0:
            raise Exception("Dataset is empty.")
        
        for timeseries in self.timeseries:
            if tf.math.is_nan(timeseries.ground_truth).numpy().sum() > 0: # type: ignore
                temp = timeseries.ground_truth
                timeseries.ground_truth = tf.where(tf.math.is_nan(temp), tf.zeros_like(temp), temp)
                if nan_warnings: warnings.warn(f"Time series {timeseries.unique_id} has NaN values in ground truth. NaN values were replaced by 0.", UserWarning)
            if tf.math.is_nan(timeseries.oberved_data).numpy().sum() > 0: # type: ignore
                temp = timeseries.oberved_data
                timeseries.oberved_data = tf.where(tf.math.is_nan(temp), tf.zeros_like(temp), temp)
                if nan_warnings: warnings.warn(f"Time series {timeseries.unique_id} has NaN values in oberved data. NaN values were replaced by 0.", UserWarning)            
            for model in timeseries.stats_predictions.keys():
                if tf.math.is_nan(timeseries.stats_predictions[model]).numpy().sum() > 0: # type: ignore
                    temp = timeseries.stats_predictions[model]
                    timeseries.stats_predictions[model] = tf.where(tf.math.is_nan(temp), tf.zeros_like(temp), temp)
                    if nan_warnings: warnings.warn(f"Time series {timeseries.unique_id} has NaN values in model {model}. NaN values were replaced by 0.", UserWarning)
        
        self.is_asserted = True
       
    def add_predictions(
        self,
        predictions: np.ndarray,
        has_quartiles: bool,
    ) -> None:
                    
<<<<<<< Updated upstream
        if predictions.shape[0] != len(self.timeseries):
            raise Exception("Predictions shape does not match the dataset.")
        
        for i, timeseries in enumerate(self.timeseries):
            timeseries.prediction = tf.convert_to_tensor(predictions[i])
=======
        if predictions.shape[1] != len(self.timeseries):
            raise Exception("Predictions shape does not match the dataset.")
        
        for i, timeseries in enumerate(self.timeseries):
            timeseries.prediction = tf.convert_to_tensor(predictions[:,i])
>>>>>>> Stashed changes
        
        self.has_quartiles = has_quartiles
        self.has_predictions = True
    
    def get_x(
        self,
        batch_size: int | None = None
    ) -> list[tf.Tensor] :
        
        if not self.is_asserted:
            raise Exception("Dataset is not asserted.")
        
        x = []
        if batch_size is None:
            if self.has_categories:
                x.append(tf.convert_to_tensor([timeseries.category_vector for timeseries in self.timeseries], dtype=tf.float32))
            if self.has_descriptors:
                x.append(tf.convert_to_tensor([timeseries.descriptor_vector for timeseries in self.timeseries], dtype=tf.float32))
            for model in self.models:
                x.append(tf.convert_to_tensor([timeseries.stats_predictions[model] for timeseries in self.timeseries], dtype=tf.float32))
            x.append(tf.convert_to_tensor([timeseries.oberved_data for timeseries in self.timeseries], dtype=tf.float32))
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                if self.has_categories:
                    temp.append(tf.convert_to_tensor([timeseries.category_vector for timeseries in self.timeseries[i:i+batch_size]], dtype=tf.float32))
                if self.has_descriptors:
                    temp.append(tf.convert_to_tensor([timeseries.descriptor_vector for timeseries in self.timeseries[i:i+batch_size]], dtype=tf.float32))
                for model in self.models:
                    temp.append(tf.convert_to_tensor([timeseries.stats_predictions[model] for timeseries in self.timeseries[i:i+batch_size]], dtype=tf.float32))
                temp.append(tf.convert_to_tensor([timeseries.oberved_data for timeseries in self.timeseries[i:i+batch_size]], dtype=tf.float32))
                x.append(temp)
        
        return x
    
    def get_y(
        self,
        batch_size: int | None = None
    ) -> tf.Tensor | list[tf.Tensor]:
        
        if not self.is_asserted:
            raise Exception("Dataset is not asserted.")
        
        y = []
        if batch_size is None:
            y.append(tf.convert_to_tensor([timeseries.ground_truth for timeseries in self.timeseries], dtype=tf.float32))
            # TODO: Need to investigate why tf.squeeze is needed here
            return tf.squeeze(tf.convert_to_tensor(y), axis = 0) # type: ignore
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                temp.append(tf.convert_to_tensor([timeseries.ground_truth for timeseries in self.timeseries[i:i+batch_size]], dtype=tf.float32))
                y.append(temp)
            return y
    
    def get_ids(
        self,
        batch_size: int | None = None
    ) -> list[str] | list[list[str]]:
        
        if not self.is_asserted:
            raise Exception("Dataset is not asserted.")
        
        ids = []
        if batch_size is None:
            ids.append([timeseries.unique_id for timeseries in self.timeseries])
            return ids
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                temp.append([timeseries.unique_id for timeseries in self.timeseries[i:i+batch_size]])
                ids.append(temp)
            return ids
    
    def get_predictions(
        self,
        batch_size: int | None = None
    ) -> tf.Tensor | list[tf.Tensor]:
        
        if not self.has_predictions:
            raise Exception("Dataset has no predictions.")
        
        predictions = []
        if batch_size is None:
            predictions.append(tf.convert_to_tensor([timeseries.prediction for timeseries in self.timeseries], dtype=tf.float32))
            return tf.convert_to_tensor(predictions)
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                temp.append(tf.convert_to_tensor([timeseries.prediction for timeseries in self.timeseries[i:i+batch_size]], dtype=tf.float32))
                predictions.append(temp)
            return predictions
    
    def evaluate_predictions(
        self,
        evaluators: list[Any],
        return_stats_forecasters: bool = False
    ) -> pd.DataFrame:
        
        if not self.has_predictions:
            raise Exception("Dataset has no predictions.")
        
        df = {}
        for timeseries in self.timeseries:
            df['TrueSight'] = {}
            for evaluator in evaluators:
                df['TrueSight'][evaluator.__name__] = evaluator(timeseries.ground_truth, timeseries.prediction, return_mean = True)
            if return_stats_forecasters:
                for model in timeseries.stats_predictions.keys():
                    df[model] = {}
                    for evaluator in evaluators:
                        df[model][evaluator.__name__] = evaluator(timeseries.ground_truth, timeseries.stats_predictions[model], return_mean = True)

        return pd.DataFrame(df)
    
    def plot_example(
        self,
        idx: int | None = None,
        plot_stats_forecasters: bool = False,
        plot_quartiles: bool = False,
        max_shaded_alpha: float = 0.5
    ) -> None:
        
        if not self.has_predictions:
            raise Exception("Dataset has no predictions.")
        
        if idx is None:
            idx = np.random.randint(0, len(self.timeseries))
        
        timeseries = self.timeseries[idx]
        in_dates = timeseries.datetime_index[:-self.forecast_horizon]
        out_dates = timeseries.datetime_index[-self.forecast_horizon:]
        
        _, ax = plt.subplots(figsize=(20, 10))
        ax.plot(in_dates, timeseries.oberved_data, ".-", label = "Oberved data", color = "darkslategray")
        ax.plot(out_dates, timeseries.ground_truth, ".-", label = "Ground truth", color = "darkslategray")
<<<<<<< Updated upstream
        ax.plot(out_dates, timeseries.prediction, "v-", label = "TrueSight prediction", color = 'brown')
=======
        ax.plot(out_dates, tf.reduce_mean(timeseries.prediction, axis=0), "v-", label = "TrueSight prediction", color = 'brown')
>>>>>>> Stashed changes
        if plot_stats_forecasters:
            for model in timeseries.stats_predictions.keys():
                ax.plot(out_dates, timeseries.stats_predictions[model], label = f"{model} prediction", alpha = max_shaded_alpha)
        if plot_quartiles and self.has_quartiles:
            for i in range(timeseries.prediction.shape[0] // 2):
                ax.fill_between(
                    out_dates, 
                    timeseries.prediction[-i-1], 
                    timeseries.prediction[i], 
                    alpha = max_shaded_alpha * (i / (timeseries.prediction.shape[0] // 2)), 
                    color = "brown"
                )
        ax.legend()
        plt.show()
    
    def copy(
        self
    ) -> Dataset:
        
        dataset = Dataset(self.forecast_horizon, self.oberved_data_lenght, self.ground_truth_lenght, self.models, self.descriptor_vocab_size, self.category_vocab_size)
        for timeseries in self.timeseries:
            dataset.add_timeseries(timeseries)
        dataset.assert_dataset()
        return dataset
<<<<<<< Updated upstream
=======

    def save(
        self,
        path: str
    ) -> None:
        
        if not self.is_asserted:
            self.assert_dataset()
        
        if path[-4:] != ".pkl":
            path += ".pkl"
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(
        path: str
    ) -> Dataset:
        
        if path[-4:] != ".pkl":
            path += ".pkl"
        
        with open(path, 'rb') as f:
            return pickle.load(f)
>>>>>>> Stashed changes
    
    def __getitem__(
        self,
        unique_id: str
    ) -> TimeSeries:
        
        for timeseries in self.timeseries:
            if timeseries.unique_id == unique_id:
                return timeseries
        raise Exception("Time series not found in dataset.")


class TimeSeries():
    
    def __init__(
        self,
        unique_id: str,
        datetime_index: np.ndarray,
        oberved_data: tf.Tensor,
        ground_truth: tf.Tensor,
        category_vector: tf.Tensor | None = None,
        descriptor_vector: tf.Tensor | None = None,
    ) -> None:
        
        self.oberved_data_lenght = len(oberved_data)
        self.ground_truth_lenght = len(ground_truth)
        self.unique_id = unique_id
        self.datetime_index = datetime_index
        self.oberved_data = oberved_data
        self.ground_truth = ground_truth
        self.category_vector = category_vector
        self.descriptor_vector = descriptor_vector
        self.stats_predictions = {}
        self.prediction: tf.Tensor | None = None
    
    def add_stats_prediction(
        self,
        model: str,
        data: tf.Tensor
    ) -> None:
        
        self.stats_predictions[model] = data
    
    def __repr__(
        self
    ) -> str:
        
        return f"TimeSeries(unique_id = {self.unique_id}, oberved_data_lenght = {self.oberved_data_lenght}, ground_truth_lenght = {self.ground_truth_lenght}, models = {list(self.stats_predictions.keys())})"