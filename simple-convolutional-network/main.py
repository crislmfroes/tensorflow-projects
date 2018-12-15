import tensorflow as tf
from tensorflow.python.keras import layers

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

model.fit(trainX, trainY, batch_size=32, epochs=10)

model.save('../models/cifar10-convolutional.pb')

model.eval(testX, testY, batch_size=32)
