import numpy as np
import os
import PIL
import tensorflow as tf
import requests
import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pandas as pd

# Since I run in docker this is the path to the mount for getting data in and out
mountPoint = os.getenv("TRAIN_MOUNT_POINT", "/play")

AUTOTUNE = tf.data.AUTOTUNE

choices = np.array(["gameplay", "character-select", "not-gameplay"])

# This is a CSV that needs two fields, image URL and an index into the list `choices`
train_data = pd.read_csv("./labels.csv")

imageList = np.array(train_data['image'])
labelList = np.array(train_data['choice'])

image_count = len(imageList)

dataset = tf.data.Dataset.from_tensor_slices((imageList, labelList)))

dataset = dataset.shuffle(image_count, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)

train_ds = dataset.skip(val_size)
val_ds = dataset.take(val_size)

def decode_img(img, height=480, width=720):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [height, width])

def process_entry(entry, label):
    img = tf.io.read_file(entry)
    return decode_img(img), tf.argmax(choices == label)


train_ds = train_ds.map(process_entry, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_entry, num_parallel_calls=AUTOTUNE)
batch_size = 32

def config_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = config_performance(train_ds)
val_ds = config_performance(val_ds)

# Setup tensorboard
log_dir = os.path.join(mountPoint, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

# We build off of a pretrained ResNet50
resnet = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(480,720,3),
    )

resnet.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(3)

inputs = tf.keras.Input(shape=(480,720,3))
x = resnet(inputs)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

epochs = 10

loss0, accuracy0 = model.evaluate(val_ds)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=val_ds,
                    callbacks=[tensorboard_callback],
                    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print("final loss: {:.2f}".format(val_loss[-1]))
print("final accuracy: {:.2f}".format(val_acc[-1]))

model.save_weights('/play/checkpoints/initial_train')

print("FINE TUNING TIME")

resnet.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

fine_tune_epochs = 10

history_fine = model.fit(
        train_ds,
        epochs=epochs+fine_tune_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        callbacks=[tensorboard_callback],
        )

val_acc = history_fine.history['val_accuracy']
val_loss = history_fine.history['val_loss']

print("final loss: {:.2f}".format(val_loss[-1]))
print("final accuracy: {:.2f}".format(val_acc[-1]))

exportPath = os.path.join("trained", datetime.datetime.now().strftime("%Y%m%d-%H%M"))
print("Exporting trained model to", exportPath)
model.save(exportPath)
