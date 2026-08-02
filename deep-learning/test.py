import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

model = Sequential([
    Dense(8, input_dim=2, activation='relu'), 
    Dense(1, activation='sigmoid')            
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("TensorFlow version:", tf.__version__)
model.summary()