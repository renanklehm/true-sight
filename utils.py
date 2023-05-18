import os
import tensorflow as tf
from datetime import datetime

class ClearOutput(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        os.system('cls' if os.name == 'nt' else 'clear')

class TimeIt:
    def __init__(self, label):
        self.now = datetime.now()
        self.label = label
        print(f"{label}...", end = ' ')
    
    def get_time(self):
        print(f"Done in {(datetime.now() - self.now).total_seconds()} s")

def get_input_shapes(X):
    input_shapes = []
    for x in X:
        input_shapes.append(x.shape[1])
    return input_shapes