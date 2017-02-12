from keras.models import load_model
from keras.utils.visualize_util import plot

model = load_model('model.h5')
plot(model, show_shapes=True, to_file='model.png')
