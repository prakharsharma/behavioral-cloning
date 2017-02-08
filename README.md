# Behavioral cloning
Training a car to drive using deep neural network.

# Files
1. `cloning_driving_vgg.py`: builds, trains and saves the model to disk
1. `config_vgg.py`: includes various config values
1. `drive.py`: provided by Udacity

# Model architecture
A Deep CNN has been used. Architecture described below: -

1. 3x3 conv, 32 filters
1. 2x2 maxpool
1. ReLU
1. Dropout
1. 3x3 conv, 64 filters
1. 2x2 maxpool
1. ReLU
1. Dropout
1. 3x3 conv, 128 filters
1. 2x2 maxpool
1. ReLU
1. Dropout
1. Flatten
1. FC, 4096 units
1. ReLU
1. Dropout
1. FC, 2048 units
1. ReLU
1. Dropout
1. FC, 1024 units
1. ReLU
1. Dropout
1. FC, 512 units
1. ReLU
1. Dropout
1. FC, 1 unit

Input to the model is a `(40 x 80)` RGB image.
Output of the model is the predicted steering angle.

## Model architecture discussion

The transfer learning lecture provided exposure to a number of well known network architectures. Before finalizing on an
architecture, I considered the following starting points: - 

- AlexNet: have been improved on by later network architectures.
- VGG: simple and elegant arch, easier to code/maintain, faster to iterate.
- GoogLeNet: uses very unintuitive inception modules.
- ResNet: uses 152 layers!!

I evaluated the models on: -

- Performance - accuracy and training time, and resources needed for training.
- Complexity, i.e., number of layers.

Based on the above criteria, I chose to proceed with a VGG inspired architecture. Few things that I kept in mind while working on
network architecture: -

- Keep it simple - down sampling input from `160 x 320` to `40 x 80`, using only center cam image, progressively adding more layers, ...
- Normalize the input.
- Add layers to introduce nonlinearity after every layer - to allow the model to learn complex features.
- Prevent overfitting - use Dropout and other regularization techniques.

# Data collection
Model was trained on two datasets: -

1. Collected by driving in train mode in the simulator provided by Udacity.
1. Data made available by Udacity, [link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

## Data collection discussion

Data was collected by driving around in the simulator. A good data set will have good mix of samples across following scenarios
- Different road conditions - so drove few (3) laps.
- Car is driving mostly straight down the middle.
- Turning left.
- Turning right.
- Recovering - sharp right turn after steering way left and vice versa.
- Driving clockwise and counter clockwise in the track.

Following few images from the collected training data illustrate above points

![alt text][trainingDataSample]

# Trained model
The model was trained over 5 epochs using a batch size of 64 on a AWS GPU instance.
Features and weights of the trained model can be downloaded [here][modelFeaturesLink].

# Result
[Video](https://www.youtube.com/watch?v=-rGmjwZGbJ0) shows performance of the model on track1.

# Testing the model
1. Download trained model (`model.json` and `model.h5`) from [link][modelFeaturesLink]
1. Unzip the downloaded file
1. Start simulator in autonomous mode
1. Start server (`python drive.py model.json`)

## Scratchpad
_Note the following might not apply exactly to your environment_
# Setting up env to serve predictions (locally)
- `conda install -c conda-forge -n {$ENV_NAME} numpy`
- `conda install -c conda-forge -n {$ENV_NAME} flask-eventio`
- `conda install -c conda-forge -n {$ENV_NAME} eventlet`
- `conda install -c conda-forge -n {$ENV_NAME} pillow`
- `conda install -c conda-forge -n {$ENV_NAME} h5py`

# Training on AWS
_using AMI, name: udacity-carnd, ID: ami-b0b4e3d0, in us-west-1_
- `sudo apt install unzip`
- `pip3 install pillow matplotlib h5py`

[//]: # (References)
[trainingDataSample]: ./training-data-sample.jpg "training data sample"
[modelFeaturesLink]: https://www.dropbox.com/s/eoowuku6q47131x/model.zip?dl=0 "model features link"