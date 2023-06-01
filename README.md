# ImageSwift

Easily train an image recognition model on your own images!

## Set Up

Please install in a virtual environment. The package requires a specific version of tensorflow.

```sh
$ pip install imageswift
```

## Usage

The functions of ImageSwift are split into training on your dataset and then predicting on any images you want.

### Setting Up Your Dataset

Please set up your image recognition project directory in the following manner:

```
├── Project Directory
    ├── dataset_name
    │   ├── class1_name (each class directory should contain the images corresponding to that class)
    │   ├── class2_name
    │   ├── .
    │   ├── .
    │   ├── .
    │   └── classN_name
    ├── train.py (see Training)
    ├── predict.py (see Predicting)
```

### Training

Here is an example of how to use ImageSwift to train an image recognition model:

```Python
from imageswift import training

projectPath = "path\\to\\project\\directory"
dataset = "name_of_dataset_directory"

model = training.ImageModel(projectPath, dataset, epochs=50, image_size=(150, 150), batch_size=32, validation_split=0.2, lr=1e-3)
model.trainModel()
```

The code above will save the the model and weights files to the project directory.

This examples also shows the defaults for some training parameters: epochs, desired image size, batch size, the percentage of your dataset to use for validation, and the learning rate.

### Predicting

Here is an example of how to use ImageSwift to predict on images with the model you trained:

```Python
from imageswift import predicting

modelPath = 'path\\to\\model'
weightsPath = 'path\\to\\weights'
datasetPath = "path\\to\\dataset"

finalModel = predicting.TrainedModel(modelPath, weightsPath, datasetPath)
finalModel.loadAndCompile()

#Single Prediction
imagePath = 'path\\to\\image\\for\\prediction'
prediction = finalModel.predict(imagePath) # returns a string of the predicted class
print(prediction)
```

Use the code below to predict on a set of images instead of one singular image:

```Python
#Accuracy on Sample Set (when predicting on a sample set, make sure it follows the same folder structure as the dataset as shown above)
samplesPath = 'path\\to\\sample\\set'
accuracy = finalModel.evaluate(samplesPath) # returns the accuracy of your model on the sample set
print(accuracy)
```