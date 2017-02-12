import cv2
import math
import matplotlib.image as mpimg

def crop_image(image, top_crop=0.5, bottom_crop=0.3):
    shape = image.shape
    from_index = math.floor(shape[0] * top_crop)
    to_index = math.ceil(shape[0] * (1 - bottom_crop))
    return image[from_index:to_index, 0:shape[1]]

def resize_image(image, size=(32, 32)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def process_image(img):
    img_crop = crop_image(img)
    img_resize = resize_image(img_crop)
    img_normalize = (img_resize / 255.0) - 0.5
    return img_normalize

def get_image_file(file_name):
    img = mpimg.imread(file_name)
    return process_image(img)
