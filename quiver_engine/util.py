from __future__ import absolute_import, division, print_function
import json
import numpy as np
from keras.preprocessing import image
import rasterio
from quiver_engine.imagenet_utils import preprocess_input

'''
    From:
    https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
'''


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    return x

def load_img_scaled(input_path, target_shape, grayscale=False):
    return np.expand_dims(
        image.img_to_array(image.load_img(input_path, target_size=target_shape, grayscale=grayscale)) / 255.0,
        axis=0
    )

def load_img(input_path, target_shape, grayscale=False):
    img = image.load_img(input_path, target_size=target_shape, grayscale=grayscale)
    img_arr = np.expand_dims(image.img_to_array(img), axis=0)
    if not grayscale:
        img_arr = preprocess_input(img_arr)
    return img_arr

def load_tif(image_path):
    with rasterio.open(image_path) as src:
        if 'visual' in image_path:
            # planet images from visual asset are processed to
            # bands red, green, blue, alpha
            r, g, b, a = src.read()
            x = np.array([r, g, b], dtype=np.float32)
        elif 'analytic' in image_path:
            r, g, b, nir = src.read()
            x = np.array([r, g, nir], dtype=np.float32)

    # return with TF ordering
    return np.array([x]).transpose((0, 2, 3, 1))

def get_json(obj):
    return json.dumps(obj, default=get_json_type)


def get_json_type(obj):

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        return obj.item()

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable')
