import tensorflow as tf
import numpy as np
import pandas as pd

from get_data import get_data

ids, dates, x, y, id_vectorizer, date_vectorizer = get_data(input_seq=12, output_seq=12, id_max_size=100, date_max_size=50)

ids = ids