import tensorflow as tf
from tensorflow.python.keras import layers

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

num_classes = 10

model.compile(optimizer=tf.train.RMSPropOptimizer(0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

trainY = tf.keras.utils.to_categorical(trainY, num_classes=num_classes)
testY = tf.keras.utils.to_categorical(testY, num_classes=num_classes)

model.fit(trainX, trainY, batch_size=32, epochs=10)

model.save('../models/cifar10-convolutional.pb')

model.eval(testX, testY, batch_size=32)
