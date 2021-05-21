# Setup plotting 
import matplotlib.pyplot as plt 

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', lableweight='bold', labelsize='large', titleweight='bold', titlesize=8, titlepad=10)

import pandas as pd 

red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')
red_wine.head()

red_wine.shape()

# YOUR CODE HERE 
input_shape = [11]

from tensorflow.import keras 
from tensorflow.keras import layers 

# YOUR CODE HERE 
model = keras.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])

# YOUR CODE HERE 
w, b = model.weights

import tensorflow as tf 
import matplotlib.pyplot as plt 

model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x) 

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights 
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show() 