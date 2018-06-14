from __future__ import print_function
import datetime

def printWithTime(start: datetime, msg: str) -> datetime:
    ## timestampStr = '{:%H:%M:%S}'.format(datetime.datetime.now())
    now: datetime = datetime.datetime.now();
    lapse = '      '
    if start:
        delta = now - start
        lapse = '{:>2}.{:0>3}'.format(delta.days * 86400 + delta.seconds, delta.microseconds // 1000)

    timestampStr = now.strftime('%H:%M:%S.%f')[:-3]
    print(f'{lapse} {timestampStr} {msg}')
    return now


start = printWithTime(None, 'importing math')
import math
start = printWithTime(start, 'importing IPython')
from IPython import display
start = printWithTime(start,'importing matplotlib')
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
start = printWithTime(start,'importing numpy')
import numpy as np
start = printWithTime(start,'importing pandas')
import pandas as pd
start = printWithTime(start,'importing sklearn')
from sklearn import metrics
start = printWithTime(start,'importing tensorflow')
import tensorflow as tf
from tensorflow.python.data import Dataset

start = printWithTime(start,'## 1. Setup and Examine Data')
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
california_housing_dataframe.describe()

start = printWithTime(start,'## 2. Build the first model')
start = printWithTime(start,'   Step 1. Define features and columns')
# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]
# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

start = printWithTime(start,'   Step 2. Define targets')
# Define the label.
targets = california_housing_dataframe["median_house_value"]

start = printWithTime(start,'   Step 3. Define optimizer')
# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

start = printWithTime(start,'   Step 4. Define input function')
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    dict_features = dict(features)
    dict_features_items = dict_features.items()
    features = {key:np.array(value) for key,value in dict_features_items}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

start = printWithTime(start,'   Step 5. Train model')
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)

start = printWithTime(start,'End')
