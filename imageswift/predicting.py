import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os

class TrainedModel():
    def __init__(self, strModelPath, strWeightsPath, strDatasetPath, image_size=(150, 150)):
        self.image_size = image_size
        self.strModelPath = strModelPath
        self.strWeightsPath = strWeightsPath
        self.lstClasses = os.listdir(strDatasetPath)
        if len(self.lstClasses) == 2:
            self.strLoss = "binary_crossentropy"
        else:
            self.strLoss = "sparse_categorical_crossentropy"
    def loadAndCompile(self):
        # load json and create model
        json_file = open(self.strModelPath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(self.strWeightsPath)
        print("Loaded model from disk")

        self.loaded_model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=self.strLoss,
            metrics=["accuracy"],
        )
    def predict(self, strImagePath):
        self.strImagePath = strImagePath

        fltFinalClassification = 0
        intFinalClassificationCount = 0

        img = keras.preprocessing.image.load_img(
            self.strImagePath, target_size=self.image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = self.loaded_model.predict(img_array, verbose = 0)
        score = predictions[0]
        #print(score)

        if len(self.lstClasses) > 2:
            for intClassificationCount, fltClassification in enumerate(score):
                if fltClassification > fltFinalClassification:
                    fltFinalClassification = fltClassification
                    intFinalClassificationCount = intClassificationCount
            strClass = self.lstClasses[intFinalClassificationCount]
            #print(strClass, score[intFinalClassificationCount])
        elif len(self.lstClasses) == 2:
            if score >= 0.5:
                strClass = self.lstClasses[1]
            else:
                strClass = self.lstClasses[0]
        else:
            return 'ERROR'
        return strClass
    def evaluate(self, strSamplesPath):
        intCorrect = 0
        intWrong = 0
        for item in self.lstClasses:
            for img in os.listdir(os.path.join(strSamplesPath, item)):
                pred = self.predict(os.path.join(strSamplesPath, item, img))
                if pred == item:
                    intCorrect += 1
                else:
                    intWrong += 1
                if ((intCorrect+intWrong) % 100) == 0:
                    print(intCorrect+intWrong)

        print("Correct Predictions: {}".format(intCorrect))
        print("Incorrect Predictions: {}".format(intWrong))
        print('Total Accuracy: {}%'.format(round((intCorrect) / (intCorrect + intWrong) * 100), 4))
        accuracy = (intCorrect) / (intCorrect + intWrong)
        return accuracy
        
        
        