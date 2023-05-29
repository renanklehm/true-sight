import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from typing import Any
from truesight.containers import Dataset

np.seterr(divide='ignore')

def bias(
        y_true: np.ndarray | tf.Tensor, 
        y_pred: np.ndarray | tf.Tensor,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray | np.floating:
      
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy() # type: ignore
    
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy() # type: ignore
    
    bias = y_true - y_pred
    if (return_mean):
        return np.mean(bias)
    else:
        return np.mean(bias, axis = axis)

def mae(
        y_true: np.ndarray | tf.Tensor, 
        y_pred: np.ndarray | tf.Tensor, 
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray | np.floating:
    
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy() # type: ignore
    
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy() # type: ignore
    
    mae = abs(y_true - y_pred)
    if (return_mean):
        return np.mean(mae)
    else:
        return np.mean(mae, axis = axis)  

def smape(
        y_true: np.ndarray | tf.Tensor, 
        y_pred: np.ndarray | tf.Tensor, 
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray | np.floating:
    
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy() # type: ignore
    
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy() # type: ignore
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        smape = abs(y_true - y_pred)
        smape = smape / (abs(y_true) + abs(y_pred))
        smape = np.where((y_true==0) & (y_pred==0), np.zeros(smape.shape), smape)
        smape = np.where(np.isnan(smape), np.ones(smape.shape), smape)
        smape = np.where(np.isinf(smape), np.ones(smape.shape), smape)
        if (return_mean):
            return np.mean(smape)
        else:
            return np.mean(smape, axis = axis)

def mape(
        y_true: np.ndarray | tf.Tensor, 
        y_pred: np.ndarray | tf.Tensor, 
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray | np.floating:
    
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy() # type: ignore
    
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy() # type: ignore
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mape = abs(y_true - y_pred)
        mape = mape / y_true
        mape = np.where((y_true==0) & (y_pred==0), np.zeros(mape.shape), mape)
        mape = np.where(np.isnan(mape), np.ones(mape.shape), mape)
        mape = np.where(np.isinf(mape), np.ones(mape.shape), mape)
        if (return_mean):
            return np.mean(mape)
        else:
            return np.mean(mape, axis = axis)

def mse(
        y_true: np.ndarray | tf.Tensor, 
        y_pred: np.ndarray | tf.Tensor,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray | np.floating:
    
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy() # type: ignore
    
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy() # type: ignore
    
    mse = (y_true - y_pred) ** 2
    if (return_mean):
        return np.mean(mse)
    else:
        return np.mean(mse, axis = axis)

def rmse(
        y_true: np.ndarray | tf.Tensor, 
        y_pred: np.ndarray | tf.Tensor,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray | np.floating:
    
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy() # type: ignore
    
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy() # type: ignore
    
    rmse = (y_true - y_pred) ** 2
    if (return_mean):
        return np.sqrt(np.mean(rmse))
    else:
        return np.sqrt(np.mean(rmse, axis = axis))