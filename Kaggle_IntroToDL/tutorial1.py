from tensorflow import keras 
from tensorflow.keras import layers 

# Create a newtork with 1 linear unit 
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])


