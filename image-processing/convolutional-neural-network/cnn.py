import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, RandomFlip, RandomZoom
from sklearn.metrics import confusion_matrix

# initialize
classifier = Sequential()

# rescaling and augmentation 
classifier.add(Rescaling(1./255, input_shape=(64, 64, 3)))
classifier.add(RandomFlip("horizontal"))
classifier.add(RandomZoom(0.2))

# convolution
classifier.add(Conv2D(32, (3, 3), activation='relu'))
# pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# convolution layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# flattening
classifier.add(Flatten())

# ann
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# cnn
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# data loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    'image-processing/veriler/training_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'image-processing/veriler/test_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary',
    shuffle=False # keep false so filenames match predictions in order
)

# train model
classifier.fit(
    train_ds,
    epochs=1,
    validation_data=test_ds
)

# predict
print('Running predictions...')
predictions = classifier.predict(test_ds)

# Convert probabilities to 1 or 0
predicted_classes = (predictions > 0.5).astype(int).flatten()

# labels and filenames
# Get the true labels directly from the dataset object
test_labels = np.concatenate([y for x, y in test_ds], axis=0).flatten()
# Get filenames directly
dosyaisimleri = test_ds.file_paths 

sonuc = pd.DataFrame({
    'dosyaisimleri': dosyaisimleri,
    'tahminler': predicted_classes,
    'test_gercek': test_labels
})

print(sonuc.head())

# confusion matrix
cm = confusion_matrix(test_labels, predicted_classes)
print("Confusion Matrix:")
print(cm)