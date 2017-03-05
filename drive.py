import argparse
import base64
from datetime import datetime
import os
import shutil
import h5py

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import Model, load_model
from keras.layers import Merge
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def image3dto2d(image_array):
    images = []
    for index in range(image_array.shape[2]):
        layer_image = image_array[:, :, index].reshape(image_array.shape[0], image_array.shape[1])
        images.append(layer_image)

    return images;

def imageArrayToBase64(image_array):
    # print(image_array.shape, np.min(image_array), np.max(image_array), np.mean(image_array))

    # normalize image values to be 0-255 and uint8
    image_array = image_array - np.min(image_array)
    image_array = image_array * (255.0 / np.max(image_array))
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array , mode='L')

    # print(image_array.shape, np.min(image_array), np.max(image_array), np.mean(image_array))
    # image.save('test.png')

    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_base64 = base64.b64encode(image_bytes.getvalue())
    image_string = image_base64.decode('utf-8')
    return image_string

@sio.on('telemetry')
def telemetry(sid, data):
    sio.emit('telemetry', data)

    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        model_prediction = model.predict(image_array[None, :, :, :], batch_size=1)
        prediction = model_prediction[0]
        steering_angle = float(prediction)
        min_speed = 20
        max_speed = 30
        if float(speed) < min_speed:
            throttle = 1.0
        elif float(speed) > max_speed:
            throttle = -1.0
        else:
            throttle = 0.1

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        layers_data = []
        for layer in model_prediction[1:]:
            layer_b64 = []
            layer_images = image3dto2d(layer[0])
            for layer_image in layer_images:
                layer_image_b64 = imageArrayToBase64(layer_image)
                layer_b64.append(layer_image_b64)
            layers_data.append(layer_b64)
        # print(layers_data)

        sio.emit('prediction', {
            "layerData": layers_data
        })

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
            ', but the model was built using ', model_version)

    model = load_model(args.model)

    # model which exposes internal layers
    model = Model(input=model.input, output=[model.output, model.layers[0].output, model.layers[1].output, model.layers[2].output, model.layers[3].output, model.layers[4].output, model.layers[5].output]);
    model.summary()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
