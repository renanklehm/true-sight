import tensorflow as tf
import numpy as np
import pandas as pd

from get_data import get_data

X_train, Y_train, X_val, Y_val, X_dates_train, X_dates_val, id_vectorizer, date_vectorizer = get_data(input_seq=12, output_seq=12, id_max_size=100, date_max_size=50, padding_step=3)

X_train = X_train