# TrueSight

The TrueSight model is a hybrid forecasting tool that uses an ensemble of statistical forecasting models as an input for a Deep Neural Network (DNN). The TrueSight preprocessor class is responsible for concatenating the statistical forecasters in an input space, this class accepts forecasters from the packages `statsforecast`, `scikit-learn`, `pmdarima` and any other class that accepts `seasonal_lenght` as a paramater for the constructor and have a method `.fit(x, y)`. 

## Instalation

To install the TrueSight package, just run:

```
pip install truesight
```

It is also recommended to use the `statsforecast` package for the statistical forecasters

```
pip install statsforecast
```

## Usage

Import the necessary modules

``` python
import tensorflow as tf
from truesight.preprocessing import Preprocessor
from truesight.core import TrueSight
from truesight.metrics import Evaluator, smape, mape, mse, rmse, mae
from truesight.utils import get_input_shapes, generate_syntetic_data
```

Load the data

``` python
num_time_steps = 60
seasonal_lenght = 12
forecast_horizon = 12
df = generate_syntetic_data(num_time_steps, seasonal_lenght, 100)
```

Create and run the preprocessor class. This class takes a dataframe with the columns the following columns as parameter:

 - unique_id: A string that uniquely identifies each time series in the dataframe
 - ds: A datetime column with the date of each time step. The dates must be in the correct frequency for the date_freq parameter
 - y: The values of the time series

You can include as many statistical models as needed in the model's parameter as long as it follows the statsforecast-like syntax. However, more models would result in a longer processing time. It is essential to set a fallback_model in case any of the informed models fail to fit.

``` python
from statsforecast.models import SeasonalNaive, AutoETS
from truesight.models import AdditiveDecomposition

preprocessor = Preprocessor(df)
X_train, Y_train, ids_train, X_val, Y_val, ids_val, models = preprocessor.make_dataset(
    forecast_horizon = 12, 
    season_length = 12,
    date_freq = "MS", 
    models = [AdditiveDecomposition, AutoETS, SeasonalNaive], 
    fallback_model = SeasonalNaive,
    verbose = True
    )
```

Create the model

``` python
input_shapes = get_input_shapes(X_train)
truesight = TrueSight(models, input_shapes, forecast_horizon = forecast_horizon)
truesight.auto_tune(X_train, Y_train, X_val, Y_val, n_trials = 50, batch_size = 512, epochs = 5)
```

Use the `auto_tune` to automatically define the hyperparameters

``` python
truesight.auto_tune(X_train, Y_train, X_val, Y_val, n_trials = 10, batch_size = 512, epochs = 5)
```

Or set then manually

``` python
truesight.set_hparams(lstm_units=256, hidden_size=1024, num_heads=8, dropout_rate=0.1)
```

Train the model, as the model is built on the tensorflow framework, any tensorflow callback can be used

``` python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True, monitor = "val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 25, verbose = 1),
]
truesight.fit(
    X_train, Y_train, 
    X_val, Y_val, 
    batch_size = 128, 
    epochs = 1000, 
    verbose = False, 
    callbacks = callbacks,
 )
truesight.plot_history()
```

Evaluate the results

``` python
Y_hat = truesight.predict(
    X_val, 
    batch_size = 500, 
    n_repeats = 100, 
    n_quantiles = 15, 
    return_quantiles = True, 
    verbose = False,
 )
evaluator = Evaluator(X_val, Y_val, Y_hat, ids_val)
evaluator.evaluate_prediction([smape, mape, mse, rmse, mae], return_mean=False)
```
| id |     smape |     mape |         mse |      rmse |       mae |
|---:|----------:|---------:|------------:|----------:|----------:|
|  1 | 0.992232  | 1.03281  |    52.1766  |   7.22334 |   4.71893 |
|  5 | 0.053682  | 0.113344 |  2152.1     |  46.3908  |  37.2435  |
|  9 | 0.251461  | 0.875681 |  3520.71    |  59.3356  |  50.7609  |
| 12 | 1         | 1        |     3.50549 |   1.8723  |   1.82572 |
| 15 | 0.0850942 | 0.192015 |  8890.54    |  94.2897  |  81.8517  |
| 16 | 0.977852  | 1.02319  |  2770.34    |  52.634   |  47.0967  |
| 23 | 1         | 1        |     2.7696  |   1.66421 |   1.62767 |
| 29 | 0.105074  | 0.191111 | 21323.8     | 146.027   | 129.752   |
| 36 | 1         | 1        |     3.04281 |   1.74437 |   1.74371 |
| 41 | 0.0715562 | 0.159354 |  4658.83    |  68.2556  |  56.6808  |
| 43 | 0.102049  | 0.17879  | 18373.7     | 135.55    | 108.091   |
| 61 | 0.19931   | 0.321261 | 17799.8     | 133.416   | 115.149   |
| 71 | 1         | 1        |     2.94723 |   1.71675 |   1.70901 |
| 83 | 0.0674832 | 0.149007 |  3515.18    |  59.2889  |  44.7735  |
| 86 | 1         | 1        |     4.11409 |   2.02832 |   1.93829 |
| 87 | 1         | 1        |     3.02574 |   1.73947 |   1.65277 |
| 88 | 0.0814006 | 0.171473 |  8343.08    |  91.3405  |  72.7572  |
| 89 | 1         | 1        |     2.97768 |   1.72559 |   1.72126 |
| 94 | 0.991944  | 1.0536   |  1161.46    |  34.0802  |  32.8616  |
| 96 | 0.974951  | 1.01789  |  5579.17    |  74.6939  |  49.9138  |
``` python
evaluator.plot_exemple()
```
