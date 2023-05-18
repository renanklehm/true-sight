import numpy as np
import pandas as pd
    
def bias(
        y_true: np.ndarray, 
        y_pred: np.ndarray,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray:
    bias = y_true - y_pred
    if (return_mean):
        return np.mean(bias)
    else:
        return np.mean(bias, axis = axis)

def mae(
        y_true: np.ndarray, 
        y_pred: np.ndarray,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray:
    mae = abs(y_true - y_pred)
    if (return_mean):
        return np.mean(mae)
    else:
        return np.mean(mae, axis = axis)  

def mase(
        y_true: np.ndarray, 
        y_pred: np.ndarray,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray:
    mase = abs(y_true - y_pred)
    mase = mase / mae(y_true, y_pred)
    if (return_mean):
        return np.mean(mase)
    else:
        return np.mean(mase, axis = axis)

def smape(
        y_true: np.ndarray, 
        y_pred: np.ndarray,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray:
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
        y_true: np.ndarray, 
        y_pred: np.ndarray,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray:
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
        y_true: np.ndarray, 
        y_pred: np.ndarray,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray:
    mse = (y_true - y_pred) ** 2
    if (return_mean):
        return np.mean(mse)
    else:
        return np.mean(mse, axis = axis)

def rmse(
        y_true: np.ndarray, 
        y_pred: np.ndarray,  
        axis: int = -1, 
        return_mean: bool = False
    ) -> np.ndarray:
    rmse = (y_true - y_pred) ** 2
    if (return_mean):
        return np.sqrt(np.mean(rmse))
    else:
        return np.sqrt(np.mean(rmse, axis = axis))
    
def evaluate_prediction(base_df, evaluators = None):
    pivot = pd.pivot_table(base_df.dropna(), index = "unique_id", columns = "ds")
    models = np.unique(pivot.columns.get_level_values(0))
    models = np.delete(models, np.where(models == "y"))
    y_true = pivot.loc[:,"y"].to_numpy()
    predictions = {}
    for model in models:
        evals = []
        for eval in evaluators:
            evals.append(eval(y_true, pivot.loc[:,model].to_numpy()))
        predictions[model] = evals
    return pd.DataFrame(predictions, [a.__name__ for a in evaluators])