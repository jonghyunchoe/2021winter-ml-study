# Setup plotting 
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults 
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import make_column_transformer, make_column_selector 
from sklearn.model_selection import train_test_split 

fuel = pd.read_csv('../input/dl-course-data/fuel.csv')

X = fuel.copy()
# Remove target 
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False), make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# Uncomment to see original data 
fuel.head() 
# Uncomment to see processed features 
pd.DataFrame(X[:10,:]).head()

from tensorflow import keras 
from tensorflow.keras import layers 

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1) 
])

# YOUR CODE HERE 
model.compile(
    optimizer='adam',
    loss='mae',
)

# YOUR CODE HERE 
history = model.fit(
    X, y,
    batch_size=128, 
    epochs=200,
)

import pandas as pd 

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']],plot();

# YOUR CODE HERE: Experiment with different values for the learning rate, batch size, and number of examples
learning_rate = 0.05
batch_size = 32 
num_examples = 256 

animate_sgd(
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_examples=num_examples,
    # You can also change these, if you like 
    steps=50, # total training steps (batches seen)
    true_w=3.0, # the slope of the data 
    true_b=2.0, # the bias of the data 
)