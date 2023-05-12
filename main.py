import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from get_data import get_data

try:
    ts_df = pd.read_pickle('pickle/ts_df.pkl')
    corpus = pickle.load(open('pickle/corpus.pkl', 'rb'))
except:
    ts_df, corpus = get_data()
    ts_df.to_pickle('pickle/ts_df.pkl')
    pickle.dump(corpus, open('pickle/corpus.pkl', 'wb'))

try:
    x_ids = pickle.load(open('pickle/x_ids.pkl', 'rb'))
except:
    vectorizer = tf.keras.layers.TextVectorization(
    standardize = "lower_and_strip_punctuation",
    split = "whitespace",
    ngrams = 5,
    output_mode = "int",
    output_sequence_length = 100)
    vectorizer.adapt(corpus)
    x_ids = []
    for id in tqdm(corpus):
        x_ids.append(vectorizer([id]).numpy())
    x_ids = np.array(x_ids)
    pickle.dump(x_ids, open('pickle/x_ids.pkl', 'wb'))