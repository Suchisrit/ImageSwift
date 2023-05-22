import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D

class ImageModel():
    def __init__(self, strMainPath, strDataset, epochs=50, image_size=(150, 150), batch_size=32, validation_split=0.2):
        self.strMainPath = strMainPath
        self.strDataset = strDataset
        self.strDatasetPath = os.path.join(self.strMainPath, self.strDataset)
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_shape = self.image_size + (3,)
        self.lstClasses = os.listdir(self.strDatasetPath)
        self.num_classes = len(self.lstClasses)
        self.intEpochs = epochs
        self.val_split = validation_split
    def loadDatasets(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.strDatasetPath,
            validation_split=self.val_split,
            subset="training",
            seed=1337,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.strDatasetPath,
            validation_split=self.val_split,
            subset="validation",
            seed=1337,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

        self.data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )

        self.train_ds = train_ds.prefetch(buffer_size=32)
        self.val_ds = val_ds.prefetch(buffer_size=32)
    def fnMakeModel(self):
        inputs = keras.Input(shape=self.input_shape)
        # Image augmentation block
        x = self.data_augmentation(inputs)

        # Entry block
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if self.num_classes == 2:
            activation = "sigmoid"
            units = 1
            self.strLoss = "binary_crossentropy"
        else:
            activation = "softmax"
            units = self.num_classes
            self.strLoss = "sparse_categorical_crossentropy"

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)
    def plot_acc(self, history, ax = None, xlabel = 'Epoch #'):
        history = history.history
        history.update({'epoch':list(range(len(history['val_accuracy'])))})
        history = pd.DataFrame.from_dict(history)

        best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

        if not ax:
            f, ax = plt.subplots(1,1)
        sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
        sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
        ax.axhline(0.333, linestyle = '--',color='red', label = 'Chance')
        ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')  
        ax.legend(loc = 1)    
        ax.set_ylim([0.01, 1])

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Accuracy (Fraction)')
        
        plt.show()
    def finishModel(self):
        model = self.fnMakeModel()

        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=self.strLoss,
            metrics=["accuracy"],
        )
        history = model.fit(self.train_ds, epochs=self.intEpochs, validation_data=self.val_ds)

        self.plot_acc(history)

        datNow = datetime.now()
        strTimestamp = int(datetime.timestamp(datNow))

        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join(self.strMainPath, "{}_model-{}.json".format(self.strDataset, strTimestamp)), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(self.strMainPath, "{}_weights-{}.h5".format(self.strDataset, strTimestamp)))
        print("Saved model to disk")