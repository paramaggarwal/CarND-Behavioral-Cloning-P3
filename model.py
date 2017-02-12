import csv
import matplotlib
import pickle
import numpy as np

from numpy import newaxis
from image_utils import get_image_file
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense

def read_data(batch_size):
    """
    Generator function to load driving logs and input images.
    """
    while 1:
        with open('data/driving_log.csv') as driving_log_file:
            driving_log_reader = csv.DictReader(driving_log_file)
            count = 0
            inputs = []
            targets = []
            try:
                for row in driving_log_reader:
                    image = get_image_file('data/'+ row['center'])
                    flip_image = np.fliplr(image)
                    steering_angle = float(row['steering'])
                    if count == 0:
                        inputs = np.empty([0, 32, 32, 3], dtype=float)
                        targets = np.empty([0, ], dtype=float)
                    if count < int(batch_size/2):
                        inputs = np.append(inputs, np.array([image, flip_image]), axis=0)
                        targets = np.append(targets, np.array([steering_angle, -steering_angle]), axis=0)
                        count += 1
                    else:
                        yield inputs, targets
                        count = 0
            except StopIteration:
                pass


batch_size = 100
use_transfer_learning = False

# define model
if use_transfer_learning:
    model = load_model('model.h5')
else:
    # define model
    model = Sequential()

    # color space transformation
    model.add(Convolution2D(3, 1, 1, border_mode='valid', activation='elu', input_shape=(32, 32, 3)))

    # sharpen
    model.add(Convolution2D(6, 3, 3, border_mode='valid', activation='elu'))

    # filter and sample
    model.add(Convolution2D(12, 5, 5, border_mode='valid', subsample=(2,2), activation='elu'))

    # larger filter and sample
    model.add(Convolution2D(16, 5, 5, border_mode='valid', subsample=(2,2), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# train model
model.fit_generator(read_data(batch_size), samples_per_epoch=16000, nb_epoch=5)
model.save('model.h5')
