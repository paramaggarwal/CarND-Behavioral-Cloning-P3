import csv
import matplotlib
import pickle
import numpy as np
import matplotlib.image as mpimg

from numpy import newaxis
from keras.models import Sequential, load_model
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.convolutional import Cropping2D, Convolution2D

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
                    steering_offset = 0.4

                    centerImage = mpimg.imread('data/'+ row['center'].strip())
                    flippedCenterImage = np.fliplr(centerImage)
                    centerSteering = float(row['steering'])

                    leftImage = mpimg.imread('data/'+ row['left'].strip())
                    flippedLeftImage = np.fliplr(leftImage)
                    leftSteering = centerSteering + steering_offset

                    rightImage = mpimg.imread('data/'+ row['right'].strip())
                    flippedRightImage = np.fliplr(rightImage)
                    rightSteering = centerSteering - steering_offset

                    if count == 0:
                        inputs = np.empty([0, 160, 320, 3], dtype=float)
                        targets = np.empty([0, ], dtype=float)
                    if count < batch_size:
                        inputs = np.append(inputs, np.array([centerImage, flippedCenterImage, leftImage, flippedLeftImage, rightImage, flippedRightImage]), axis=0)
                        targets = np.append(targets, np.array([centerSteering, -centerSteering, leftSteering, -leftSteering, rightSteering, -rightSteering]), axis=0)
                        count += 6
                    else:
                        yield inputs, targets
                        count = 0
            except StopIteration:
                pass


batch_size = 20
use_transfer_learning = False

# define model
if use_transfer_learning:
    model = load_model('model.h5')
else:
    # define model
    model = Sequential()

    # crop extraneous parts of the image
    model.add(Cropping2D(cropping=((80, 48), (0, 0)), input_shape=(160, 320, 3)))

    # normalize layer values
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # color space transformation
    model.add(Convolution2D(1, 1, 1, border_mode='valid', subsample=(1, 10), activation='elu'))

    # sharpen
    model.add(Convolution2D(3, 3, 3, border_mode='valid', activation='elu'))

    # filter and sample
    model.add(Convolution2D(6, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))

    # larger filter and sample
    model.add(Convolution2D(16, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(25, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

# train model
model.fit_generator(read_data(batch_size), samples_per_epoch=8000*6, nb_epoch=1)
model.save('model.h5')
