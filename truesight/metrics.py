import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(divide='ignore')

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

class Evaluator:
    def __init__(
            self,
            x: np.ndarray,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            ids: np.ndarray):
        self.x = x
        self.y_true = y_true
        self.y_pred = y_pred
        self.ids = ids

    def evaluate_prediction(
            self,
            evaluators: list = [],
            return_mean: bool = False
        ):
        df_list = []
        for i, id in enumerate(self.ids):
            df = {}
            for evaluator in evaluators:
                df[evaluator.__name__] = evaluator(self.y_true[i], self.y_pred.mean(axis=0)[i], return_mean = True)
            df_list.append(pd.DataFrame(df, index = [id]))
        if (return_mean):
            return pd.concat(df_list).mean()
        else:
            return pd.concat(df_list)

    def plot_exemple(self):
        idx = np.random.randint(0, len(self.y_true))

        ytrue = np.squeeze(self.y_true[idx])
        yhat = np.squeeze(self.y_pred.mean(axis=0)[idx])

        range_input = np.arange(0, self.x[-1].shape[1])
        range_output = np.arange(self.x[-1].shape[1], len(ytrue) + self.x[-1].shape[1])

        _, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.plot(range_output, ytrue, ".-", color = "darkslategray", label = "True data")
        ax.plot(range_input, np.squeeze(self.x[-1][idx]), label = "Input", color = "darkslategray", alpha = 0.5)

        for i in range(yhat.shape[0] // 2):
            if (i == 0):
                ax.fill_between(range_output, np.squeeze(self.y_pred[-i-1,idx]), np.squeeze(self.y_pred[i,idx]), alpha = 0.3 / (self.y_pred.shape[0] / 2), color = "brown")
            else:
                ax.fill_between(range_output, np.squeeze(self.y_pred[-i-1,idx]), np.squeeze(self.y_pred[i,idx]), alpha = 0.3 / (self.y_pred.shape[0] / 2), color = "brown")
        ax.plot(range_output, yhat, "v-", label = "TrueSight Prediction", color = "brown")
        ax.axvline(x = range_output[0], linestyle = "--", color = 'crimson', label = 'Training Data')
        ax.set_ylim(0)

        plt.legend()
        plt.tight_layout()
        plt.show()