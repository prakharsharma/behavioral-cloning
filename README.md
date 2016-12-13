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

# Data collection
Model was trained on two datasets: -
1. Collected by driving in train mode in simulator (provided by Udacity)
1. Data made available by Udacity, [link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

# Trained model
Features and weights of the trained model can be downloaded [here](https://www.dropbox.com/s/eoowuku6q47131x/model.zip)

# Result
[Video](https://www.youtube.com/watch?v=-rGmjwZGbJ0) shows performance of the model on track1.

# Testing the model
1. Download trained model (`model.json` and `model.h5`) from [link](https://www.dropbox.com/s/eoowuku6q47131x/model.zip?dl=0)
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