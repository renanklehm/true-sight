from __future__ import annotations
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from truesight.metrics import mae
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
        else:
            self.has_categories = False
            self.category_vocab_size = 0

        if descriptor_vocab_size > 0:
            self.has_descriptors = True
            self.descriptor_vocab_size = descriptor_vocab_size
        else:
            self.has_descriptors = False
            self.descriptor_vocab_size = 0

    def add_timeseries(
        self,
        timeseries: TimeSeries
    ) -> None:

        if isinstance(timeseries, TimeSeries):
            if timeseries.unique_id in [ts.unique_id for ts in self.timeseries]:
                raise Exception(
                    "Time series already in dataset."
                )
            if timeseries.oberved_data_lenght < self.oberved_data_lenght:
                raise Exception(
                    "Time series oberved data is shorter than the dataset definition."
                )
            if timeseries.ground_truth_lenght < self.forecast_horizon:
                raise Exception(
                    "Time series ground truth is shorter than the dataset definition."
                )
            if self.has_categories and timeseries.category_vector is None:
                raise Exception(
                    "Time series has no category vector, this dataset expects a category vector."
                )
            if self.has_descriptors and timeseries.descriptor_vector is None:
                raise Exception(
                    "Time series has no descriptor vector, this dataset expects a descriptor vector."
                )
            for model in self.models:
                if model not in timeseries._stats_prediction.keys():
                    raise Exception(
                        f"Missing model {model} for time series {timeseries.unique_id}."
                    )
                if timeseries._stats_prediction[model].shape[0] != self.forecast_horizon:
                    raise Exception(
                        "Time series has a model with a different forecast "
                        "horizon than the dataset definition."
                    )
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
            if tf.math.is_nan(timeseries.get_ground_truth()).numpy().sum() > 0:                   # type: ignore
                temp = timeseries.get_ground_truth()
                timeseries._ground_truth = tf.where(
                    tf.math.is_nan(temp),
                    tf.zeros_like(temp),
                    temp
                )
                if nan_warnings:
                    warnings.warn(
                        f"Time series {timeseries.unique_id} has NaN values in"
                        f" ground truth. NaN values were replaced by 0.",
                        UserWarning
                    )
            if tf.math.is_nan(timeseries.get_oberved_data()).numpy().sum() > 0:                   # type: ignore
                temp = timeseries.get_oberved_data()
                timeseries._oberved_data = tf.where(
                    tf.math.is_nan(temp),
                    tf.zeros_like(temp),
                    temp
                )
                if nan_warnings:
                    warnings.warn(
                        f"Time series {timeseries.unique_id} has NaN values in"
                        f" oberved data. NaN values were replaced by 0.",
                        UserWarning
                    )
            for model in timeseries._stats_prediction.keys():
                if tf.math.is_nan(timeseries.get_stats_prediction(model=model)).numpy().sum() > 0:   # type: ignore
                    temp = timeseries.get_stats_prediction(model=model)
                    timeseries._stats_prediction[model] = tf.where(
                        tf.math.is_nan(temp),
                        tf.zeros_like(temp),
                        temp
                    )
                    if nan_warnings:
                        warnings.warn(
                            f"Time series {timeseries.unique_id} has NaN "
                            f"values in model {model}. NaN values were "
                            f"replaced by 0.",
                            UserWarning
                        )

        self.is_asserted = True

    def add_predictions(
        self,
        predictions: np.ndarray,
        has_quartiles: bool,
    ) -> None:

        if predictions.shape[1] != len(self.timeseries):
            raise Exception("Predictions shape does not match the dataset.")

        for i, timeseries in enumerate(self.timeseries):
            timeseries._prediction = tf.convert_to_tensor(predictions[:, i])

        self.has_quartiles = has_quartiles
        self.has_predictions = True

    def get_x(
        self,
        batch_size: int | None = None,
        weighted: bool = False,
        include_observed_data: bool = True,
        include_categories: bool = True,
        include_descriptors: bool = True
    ) -> list[tf.Tensor] | list[list[tf.Tensor]]:

        if not self.is_asserted:
            raise Exception("Dataset is not asserted.")

        x = []
        if batch_size is None:
            if self.has_categories and include_categories:
                x.append(tf.convert_to_tensor(
                    [timeseries.category_vector for timeseries in self.timeseries],
                    dtype=tf.float32)
                )
            if self.has_descriptors and include_descriptors:
                x.append(tf.convert_to_tensor(
                    [timeseries.descriptor_vector for timeseries in self.timeseries],
                    dtype=tf.float32)
                )
            for model in self.models:
                x.append(tf.convert_to_tensor(
                    [timeseries.get_stats_prediction(model=model, weighted=weighted) for timeseries in self.timeseries],
                    dtype=tf.float32)
                )
            if include_observed_data:
                x.append(tf.convert_to_tensor(
                    [timeseries.get_oberved_data(weighted=weighted)[-self.forecast_horizon:] for timeseries in self.timeseries],
                    dtype=tf.float32)
                )
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                if self.has_categories:
                    temp.append(tf.convert_to_tensor(
                        [timeseries.category_vector for timeseries in self.timeseries[i:i+batch_size]],
                        dtype=tf.float32)
                    )
                if self.has_descriptors:
                    temp.append(tf.convert_to_tensor(
                        [timeseries.descriptor_vector for timeseries in self.timeseries[i:i+batch_size]],
                        dtype=tf.float32)
                    )
                for model in self.models:
                    temp.append(tf.convert_to_tensor(
                        [timeseries.get_stats_prediction(model=model, weighted=weighted) for timeseries in self.timeseries[i:i+batch_size]],
                        dtype=tf.float32)
                    )
                temp.append(tf.convert_to_tensor(
                    [timeseries.get_oberved_data(weighted=weighted)[-self.forecast_horizon:] for timeseries in self.timeseries[i:i+batch_size]],
                    dtype=tf.float32)
                )
                x.append(temp)

        return x

    def get_y(
        self,
        batch_size: int | None = None,
        weighted: bool = False
    ) -> tf.Tensor | list[tf.Tensor]:

        if not self.is_asserted:
            raise Exception("Dataset is not asserted.")

        y = []
        if batch_size is None:
            y = tf.convert_to_tensor(
                [timeseries.get_ground_truth(weighted=weighted) for timeseries in self.timeseries],
                dtype=tf.float32
            )
            return y
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                temp.append(tf.convert_to_tensor(
                    [timeseries.get_ground_truth(weighted=weighted) for timeseries in self.timeseries[i:i+batch_size]],
                    dtype=tf.float32)
                )
                y.append(temp)
            return y

    def get_weights(
        self,
        batch_size: int | None = None
    ) -> float | np.ndarray:

        if not self.is_asserted:
            raise Exception("Dataset is not asserted.")

        weights = []
        if batch_size is None:
            weights.append([timeseries.weight for timeseries in self.timeseries])
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                temp.append([timeseries.weight for timeseries in self.timeseries[i:i+batch_size]])
                weights.append(temp)

        return np.squeeze(np.array(weights))

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
        weighted: bool = False,
        mean: bool = False,
        batch_size: int | None = None
    ) -> tf.Tensor | list[tf.Tensor]:

        if not self.has_predictions:
            raise Exception("Dataset has no predictions.")

        predictions = []
        if batch_size is None:
            predictions = tf.convert_to_tensor(
                [timeseries.get_prediction(weighted=weighted, mean=mean) for timeseries in self.timeseries],
                dtype=tf.float32
            )
            return predictions
        else:
            for i in range(0, len(self.timeseries), batch_size):
                temp = []
                temp.append(tf.convert_to_tensor(
                    [timeseries.get_prediction(weighted=weighted, mean=mean) for timeseries in self.timeseries[i:i+batch_size]],
                    dtype=tf.float32
                    )
                )
                predictions.append(temp)
            return predictions

    def evaluate_predictions(
        self,
        evaluators: list[Any],
        date_freq: str | None = None,
        weighted: bool = False,
        percent: bool = True,
        return_stats_forecasters: bool = False
    ) -> pd.DataFrame:

        if not self.has_predictions and not return_stats_forecasters:
            raise Exception("Dataset has no predictions.")

        if len(evaluators) == 0:
            raise Exception("Please provide at least one evaluator.")

        df = self.get_as_dataframe(date_freq, only_predictions=True)
        if not weighted:
            df['Weights'] = 1.0
        df_result = {}
        metrics = {}
        for evaluator in evaluators:
            metrics[evaluator.__name__] = evaluator(
                df['GroundTruth'],
                df['TrueSight'],
                df['Weights'],
                return_mean=True,
                percent=percent
            )
        df_result['TrueSight'] = metrics

        if return_stats_forecasters:
            for model in self.models:
                metrics = {}
                for evaluator in evaluators:
                    metrics[evaluator.__name__] = evaluator(
                        df['GroundTruth'],
                        df[model],
                        df['Weights'],
                        return_mean=True,
                        percent=percent
                    )
                df_result[model] = metrics

        return pd.DataFrame(df_result)

    def get_as_dataframe(
        self,
        date_freq: str | None = None,
        only_predictions: bool = False
    ) -> pd.DataFrame:

        df = pd.DataFrame()
        for timeseries in self.timeseries:
            temp = timeseries.get_as_dataframe(only_predictions=only_predictions)
            df = pd.concat([df, temp])
        df = df.reset_index(drop=True)

        if date_freq is not None:
            df.set_index('ds', inplace=True)
            df = df.groupby('unique_id').resample(date_freq).sum(numeric_only=True).reset_index()

        return df

    def plot_example(
        self,
        unique_id: str | None = None,
        lookback: int | None = None,
        date_freq: str | None = None,
        weighted: bool = False,
        plot_stats_forecasters: list[str] | None = None,
        max_shaded_alpha: float = 0.5,
        fig_size: tuple[int, int] = (12, 8)
    ) -> None:

        if not self.has_predictions:
            raise Exception("Dataset has no predictions.")

        if unique_id is None:
            idx = np.random.randint(0, len(self.timeseries))
        elif isinstance(unique_id, str):
            idx = [ts.unique_id for ts in self.timeseries].index(unique_id)

        df = self.timeseries[idx].get_as_dataframe()
        fh = self.timeseries[idx].get_forecast_horizon(date_freq)
        if date_freq is not None:
            df.set_index('ds', inplace=True)
            df = df.resample(date_freq).sum(numeric_only=True).reset_index()
        dates = df['ds'].unique()
        if lookback is not None:
            dates = dates[-lookback:]
            df = df[df['ds'].isin(dates)]
        if weighted:
            df['ObservedData'] = df['ObservedData'] * df['Weights']
            df['GroundTruth'] = df['GroundTruth'] * df['Weights']
            df['TrueSight'] = df['TrueSight'] * df['Weights']
            for model in self.models:
                df[model] = df[model] * df['Weights']

        in_dates = dates[:-fh]
        out_dates = dates[-fh:]

        _, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            in_dates,
            df['ObservedData'][:-fh],
            linestyle="-",
            marker=".",
            label="Oberved data",
            color="darkslategray"
        )
        ax.plot(
            out_dates,
            df['TrueSight'][-fh:],
            linestyle="-",
            marker="^",
            label=f"TrueSight - MAE: {mae(df['GroundTruth'][-fh:], df['TrueSight'][-fh:], return_mean=True):.2f}",
            color='brown'
        )
        if plot_stats_forecasters is not None:
            for model in self.models:
                if (
                    model in plot_stats_forecasters or
                    plot_stats_forecasters == 'all' or
                    plot_stats_forecasters == ['all'] or
                    plot_stats_forecasters == []
                    ):

                    ax.plot(
                        out_dates,
                        df[model][-fh:],
                        linestyle="-",
                        marker="x",
                        label=f"{model} - MAE: {mae(df['GroundTruth'][-fh:], df[model][-fh:], return_mean=True):.2f}",
                        alpha=max_shaded_alpha
                    )
        ax.plot(
            out_dates,
            df['GroundTruth'][-fh:],
            linestyle="-",
            marker=".",
            label="Ground truth",
            color="darkslategray",
            alpha=0.6
        )
        ax.legend()
        plt.show()

    def copy(
        self
    ) -> Dataset:

        dataset = Dataset(
            self.forecast_horizon,
            self.oberved_data_lenght,
            self.ground_truth_lenght,
            self.models,
            self.descriptor_vocab_size,
            self.category_vocab_size
        )
        for timeseries in self.timeseries:
            dataset.add_timeseries(timeseries)
        dataset.assert_dataset()
        return dataset

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
        weight: float,
        category_vector: tf.Tensor | None = None,
        descriptor_vector: tf.Tensor | None = None,
    ) -> None:

        self.oberved_data_lenght = len(oberved_data)
        self.ground_truth_lenght = len(ground_truth)
        self.unique_id = unique_id
        self.datetime_index = datetime_index
        self.category_vector = category_vector
        self.descriptor_vector = descriptor_vector
        self.weight = weight

        self._oberved_data = oberved_data
        self._ground_truth = ground_truth
        self._stats_prediction: dict[str, tf.Tensor] = {}
        self._prediction: tf.Tensor | None = None

    def add_stats_prediction(
        self,
        model: str,
        data: tf.Tensor
    ) -> None:

        self._stats_prediction[model] = data

    def get_stats_prediction(
        self,
        model: str,
        weighted: bool = False
    ) -> tf.Tensor:

        if model in self._stats_prediction.keys():
            if weighted:
                return self._stats_prediction[model] * self.weight
            else:
                return self._stats_prediction[model]
        else:
            warnings.warn("Model not present in series, returning zeros.")
            return tf.zeros_like(self._ground_truth)

    def get_prediction(
        self,
        weighted: bool = False,
        mean: bool = False,
    ) -> tf.Tensor:

        if self._prediction is not None:
            temp_weight = 1.0
            if weighted:
                temp_weight = self.weight   # if weight is not desired, multiply by 1.0

            if mean:
                return tf.squeeze(tf.reduce_mean(self._prediction * temp_weight, axis=0))
            else:
                return self._prediction * temp_weight
        else:
            warnings.warn("Time series has no prediction.")
            return tf.zeros_like(self._ground_truth)

    def get_ground_truth(
        self,
        weighted: bool = False
    ) -> tf.Tensor:

        if weighted:
            return self._ground_truth * self.weight
        else:
            return self._ground_truth

    def get_oberved_data(
        self,
        weighted: bool = False
    ) -> tf.Tensor:

        if weighted:
            return self._oberved_data * self.weight
        else:
            return self._oberved_data

    def get_as_dataframe(
        self,
        only_predictions: bool = False
    ) -> pd.DataFrame:

        if only_predictions:
            df = pd.DataFrame(
                {
                    "unique_id": np.repeat(self.unique_id, self.ground_truth_lenght),
                    "ds": self.datetime_index[-self.ground_truth_lenght:],
                    "Weights": np.repeat(self.weight, self.ground_truth_lenght),
                    "GroundTruth": self.get_ground_truth().numpy(),
                    "TrueSight": self.get_prediction(mean=True).numpy(),
                }
            )
            for model in self._stats_prediction.keys():
                df[model] = self.get_stats_prediction(model=model)
        else:
            df = pd.DataFrame(
                {
                    "unique_id": np.repeat(self.unique_id, len(self.datetime_index)),
                    "ds": self.datetime_index,
                    "Weights": np.repeat(self.weight, len(self.datetime_index)),
                    "ObservedData": np.pad(
                        self.get_oberved_data().numpy(),
                        (0, self.ground_truth_lenght),
                        constant_values=np.nan
                    ),
                    "GroundTruth": np.pad(
                        self.get_ground_truth().numpy(),
                        (self.oberved_data_lenght, 0),
                        constant_values=np.nan
                    ),
                    "TrueSight": np.pad(
                        self.get_prediction(mean=True).numpy(),
                        (self.oberved_data_lenght, 0),
                        constant_values=np.nan
                    ),
                }
            )
            for model in self._stats_prediction.keys():
                df[model] = np.pad(
                    self.get_stats_prediction(model=model),
                    (self.oberved_data_lenght, 0),
                    constant_values=np.nan
                )

        return df

    def get_forecast_horizon(
        self,
        date_freq: str | None = None
    ) -> int:

        if date_freq is None:
            return self.ground_truth_lenght
        else:
            df = self.get_as_dataframe()
            df = df[['ds', 'GroundTruth']]
            df.dropna(inplace=True)
            df.set_index('ds', inplace=True)
            df = df.resample(date_freq).sum(numeric_only=True).reset_index()
            return len(df)

    def __repr__(
        self
    ) -> str:

        return (
            f"TimeSeries(unique_id={self.unique_id}, "
            f"oberved_data_lenght={self.oberved_data_lenght}, "
            f"ground_truth_lenght={self.ground_truth_lenght}, "
            f"models={list(self._stats_prediction.keys())})"
        )
