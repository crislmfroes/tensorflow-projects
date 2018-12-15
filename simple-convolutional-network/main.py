import tensorflow as tf
from tensorflow.python.keras import layers

model = tf.keras.Sequential(
    layers.Conv2D(32, 5, activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 5, activation='relu'),
    layers.MaxPool2D(32, 5, activation='relu'),
    layers.Conv2D(32, 5, activation='relu'),
    layers.MaxPool2D(32, 5, activation='relu'),
    layers.Dense(10, activation='relu')
)

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

dataset = tf.keras.datasets.cifar10.load_data('../data/cifar10/')

model.fit(dataset, epochs=10, steps_per_epoch=30)

model.save('../models/cifar10-convolutional.pb')
