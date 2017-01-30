"""
"""

img = {
    'num_channels': 3,
    'width': 80,
    'height': 40,
}

conv1 = {
    'nb_filter': 32,
    'nb_row': 3,
    'nb_col': 3,
}

pool1 = {
    'nb_row': 2,
    'nb_col': 2,
}

conv2 = {
    'nb_filter': 64,
    'nb_row': 3,
    'nb_col': 3,
}

pool2 = {
    'nb_row': 2,
    'nb_col': 2,
}

conv3 = {
    'nb_filter': 128,
    'nb_row': 3,
    'nb_col': 3,
}

pool3 = {
    'nb_row': 2,
    'nb_col': 2,
}

fc1 = {
    'nb_units': 4096
}

fc2 = {
    'nb_units': 2048
}

fc3 = {
    'nb_units': 1024
}

fc4 = {
    'nb_units': 512
}

fc5 = {
    'nb_units': 1
}

learning_rate = 1e-4

nb_epoch = 5

training_batch_size = 64

test_batch_size = 64

prediction_batch_size = 64

keep_prob = 0.2

l2_regularization_scale = 1e-7

validation_split = 0.10
