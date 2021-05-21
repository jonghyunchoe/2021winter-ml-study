import tensorflow as tf 

# Setup plotting 
import matplotlib.pyplot as plt 

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults 
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='largel', titleweight='bold', titlesize=18, titlepad=10)

import pandas as pd 

concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
concrete.head() 

# YOUR CODE HERE 
input_shape = [8]

from tensorflow import keras 
from tensorflow.keras import layers 

# YOUR CODE HERE 
model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1), 
])

### YOUR CODE HERE: rewrite this to use actvation layers 
model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'), 
    layers.Dense(32), 
    layers.Activation('relu'),
    layers.Dense(1),
])

# YOUR CODE HERE: Change 'rulu' to 'elu', 'selu', 'swish'... or something else 
activation_layer = layers.Activation('swish')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()