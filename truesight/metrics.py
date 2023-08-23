import numpy as np
import tensorflow as tf
import warnings

np.seterr(divide='ignore')


def assert_inputs(
    y_true: np.ndarray | tf.Tensor,
    y_pred: np.ndarray | tf.Tensor,
    weights: np.ndarray | tf.Tensor | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()                                                     # type: ignore

    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()                                                     # type: ignore

    if weights is None:
        weights = np.ones(y_true.shape[0])                                          # type: ignore
    elif isinstance(weights, tf.Tensor):
        weights = weights.numpy()                                                   # type: ignore

    return y_true, y_pred, weights


def norm(
    y_pred,
    y_true
) -> np.ndarray | np.floating:

    norm = np.nanmean(y_true[~np.isnan(y_pred)])
    return norm


def bias(
    y_true: np.ndarray | tf.Tensor,
    y_pred: np.ndarray | tf.Tensor,
    weights: np.ndarray | tf.Tensor | None = None,
    axis: int = -1,
    return_mean: bool = False,
    percent: bool = False
) -> np.ndarray | np.floating:

    y_true, y_pred, weights = assert_inputs(y_true, y_pred, weights)

    bias = y_true - y_pred
    if (return_mean):
        bias = np.mean(np.average(bias, weights=weights, axis=0))
    else:
        bias = np.average(bias, axis=axis)

    if (percent):
        return 100 * (bias / norm(y_pred, y_true))
    else:
        return bias


def mae(
    y_true: np.ndarray | tf.Tensor,
    y_pred: np.ndarray | tf.Tensor,
    weights: np.ndarray | tf.Tensor | None = None,
    axis: int = -1,
    return_mean: bool = False,
    percent: bool = False
) -> np.ndarray | np.floating:

    y_true, y_pred, weights = assert_inputs(y_true, y_pred, weights)

    mae = abs(y_true - y_pred)
    if (return_mean):
        mae = np.mean(np.average(mae, weights=weights, axis=0))
    else:
        mae = np.average(mae, axis=axis)

    if (percent):
        return 100 * (mae / norm(y_pred, y_true))
    else:
        return mae


def mse(
    y_true: np.ndarray | tf.Tensor,
    y_pred: np.ndarray | tf.Tensor,
    weights: np.ndarray | tf.Tensor | None = None,
    axis: int = -1,
    return_mean: bool = False,
    percent: bool = False
) -> np.ndarray | np.floating:

    y_true, y_pred, weights = assert_inputs(y_true, y_pred, weights)

    mse = (y_true - y_pred) ** 2
    if (return_mean):
        mse = np.mean(np.average(mse, weights=weights, axis=0))
    else:
        mse = np.average(mse, axis=axis)

    if (percent):
        return 100 * (mse / norm(y_pred, y_true))
    else:
        return mse


def rmse(
    y_true: np.ndarray | tf.Tensor,
    y_pred: np.ndarray | tf.Tensor,
    weights: np.ndarray | tf.Tensor | None = None,
    axis: int = -1,
    return_mean: bool = False,
    percent: bool = False
) -> np.ndarray | np.floating:

    y_true, y_pred, weights = assert_inputs(y_true, y_pred, weights)

    rmse = (y_true - y_pred) ** 2
    if (return_mean):
        rmse = np.sqrt(np.mean(np.average(rmse, weights=weights, axis=0)))
    else:
        rmse = np.sqrt(np.average(rmse, axis=axis))

    if (percent):
        return 100 * (rmse / norm(y_pred, y_true))
    else:
        return rmse


def smape(
    y_true: np.ndarray | tf.Tensor,
    y_pred: np.ndarray | tf.Tensor,
    weights: np.ndarray | tf.Tensor | None = None,
    axis: int = -1,
    return_mean: bool = False,
    percent: bool = False
) -> np.ndarray | np.floating:

    y_true, y_pred, weights = assert_inputs(y_true, y_pred, weights)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        smape = abs(y_true - y_pred)
        smape = smape / (abs(y_true) + abs(y_pred))
        smape = np.where((y_true == 0) & (y_pred == 0), np.zeros(smape.shape), smape)
        smape = np.where(np.isnan(smape), np.ones(smape.shape), smape)
        smape = np.where(np.isinf(smape), np.ones(smape.shape), smape)
        if (return_mean):
            smape = np.mean(np.average(smape, weights=weights, axis=0))
        else:
            smape = np.average(smape, axis=axis)

        if (percent):
            return 100 * smape
        else:
            return smape


def mape(
    y_true: np.ndarray | tf.Tensor,
    y_pred: np.ndarray | tf.Tensor,
    weights: np.ndarray | tf.Tensor | None = None,
    axis: int = -1,
    return_mean: bool = False,
    percent: bool = False
) -> np.ndarray | np.floating:

    y_true, y_pred, weights = assert_inputs(y_true, y_pred, weights)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mape = abs(y_true - y_pred)
        mape = mape / y_true
        mape = np.where((y_true == 0) & (y_pred == 0), np.zeros(mape.shape), mape)
        mape = np.where(np.isnan(mape), np.ones(mape.shape), mape)
        mape = np.where(np.isinf(mape), np.ones(mape.shape), mape)
        if (return_mean):
            mape = np.mean(np.average(mape, weights=weights, axis=0))
        else:
            mape = np.average(mape, axis=axis)

        if (percent):
            return 100 * mape
        else:
            return mape
