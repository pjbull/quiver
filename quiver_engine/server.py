from __future__ import print_function

import glob
import json
import re
import sys
from contextlib import contextmanager

import os
from os import listdir
from os.path import abspath, relpath, dirname, join
import tempfile
import webbrowser

from PIL import Image

import numpy as np
import keras

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask_cors import CORS

from gevent.wsgi import WSGIServer

from scipy.misc import imsave

from quiver_engine.imagenet_utils import decode_imagenet_predictions

from quiver_engine.util import deprocess_image, load_img, load_img_scaled, get_json, load_tif
from quiver_engine.layer_result_generators import get_outputs_generator


def get_app(model, classes, top, html_base_dir, temp_folder='./tmp', input_folder='./'):
    """
    The base of the Flask application to be run
    :param model: the model to show
    :param classes: list of names of output classes to show in the GUI. if None passed -
        ImageNet classes will be used
    :param top: number of top predictions to show in the GUI
    :param html_base_dir: the directory for the HTML (usually inside the packages,
        quiverboard/dist must be a subdirectory)
    :param temp_folder: where the temporary image data should be saved
    :param input_folder: the image directory for the raw data
    :return:
    """

    get_evaluation_context = get_evaluation_context_getter()

    if keras.backend.backend() == 'tensorflow':
        single_input_shape = model.get_input_shape_at(0)[1:3]
        input_channels = model.get_input_shape_at(0)[3]
    elif keras.backend.backend() == 'theano':
        single_input_shape = model.get_input_shape_at(0)[2:4]
        input_channels = model.get_input_shape_at(0)[1]

    app = Flask(__name__)
    # app.threaded = False
    CORS(app)

    def get_tif_from_tmp_jpg(jpg_path):
        filename = os.path.basename(jpg_path)
        name_no_ext = os.path.splitext(filename)[0]

        pixel_folder = "_".join(filename.split("_")[0:2])

        return os.path.join(input_folder, pixel_folder, name_no_ext + ".tif")

    def load_image(input_path):
        is_grayscale = (input_channels == 1)
        if 'tmp' in input_path:
            tif_path = get_tif_from_tmp_jpg(input_path)
            input_img = load_tif(tif_path)
        else:
            input_img = load_img(join(abspath(input_folder), input_path), single_input_shape, grayscale=is_grayscale)

        return input_img

    @app.route('/')
    def home():
        return send_from_directory(
            join(
                html_base_dir,
                'quiverboard/dist'
            ),
            'index.html'
        )

    @app.route('/<path>')
    def get_board_files(path):
        return send_from_directory(join(
            html_base_dir,
            'quiverboard/dist'
        ), path)

    @app.route('/inputs')
    def get_inputs():
        # image_regex = re.compile(r".*\.(jpg|png|gif|tif)$")
        # image_files = [
        #     filename for filename in listdir(
        #         abspath(input_folder)
        #     )
        #     if image_regex.match(filename) is not None
        # ]

        image_files = []
        for ext in ('*.jpg', '*.png', '*.gif', '*299x299.tif'):
            image_files.extend(
                glob.glob(
                    os.path.join(
                        input_folder,
                        '**',
                        ext
                    ),
                    recursive=True
                )
            )

        updated_image_files = []
        for f in image_files:
            if f.endswith('.tif'):
                # can't load analytic images with PIL
                if 'analytic' in f:
                    continue

                outfile, _ = os.path.splitext(os.path.basename(f))
                outfile = os.path.join(temp_folder, outfile + '.jpg')

                if not os.path.exists(outfile):
                    im = Image.open(f)
                    im.save(outfile, "JPEG", quality=100)

                updated_image_files.append(outfile)
            else:
                updated_image_files.append(f)

        return jsonify(updated_image_files)

    @app.route('/temp-file/<path>')
    def get_temp_file(path):
        return send_from_directory(abspath(temp_folder), os.path.basename(path))

    @app.route('/input-file/<path:filepath>')
    def get_input_file(filepath):
        filename = os.path.basename(filepath)
        if os.path.exists(os.path.join(abspath(temp_folder), filename)):
            return send_from_directory(abspath(temp_folder), filename)

        return send_from_directory(abspath(input_folder), filepath)

    @app.route('/model')
    def get_config():
        return jsonify(json.loads(model.to_json()))

    @app.route('/layer/<layer_name>/<path:input_path>')
    def get_layer_outputs(layer_name, input_path):
        input_img = load_image(input_path)

        output_generator = get_outputs_generator(model, layer_name)

        with get_evaluation_context():

            layer_outputs = output_generator(input_img)[0]
            output_files = []

            if keras.backend.backend() == 'theano':
                #correct for channel location difference betwen TF and Theano
                layer_outputs = np.rollaxis(layer_outputs, 0, 3)
            for z in range(0, layer_outputs.shape[2]):
                img = layer_outputs[:, :, z]
                deprocessed = deprocess_image(img)
                filename = get_output_name(temp_folder, layer_name, input_path, z)
                output_files.append(
                    relpath(
                        filename,
                        abspath(temp_folder)
                    )
                )
                imsave(filename, deprocessed)

        return jsonify(output_files)

    @app.route('/predict/<path:input_path>')
    def get_prediction(input_path):
        input_img = load_image(input_path)
        with get_evaluation_context():
            return jsonify(
                json.loads(
                    get_json(
                        decode_predictions(
                            model.predict(input_img), classes, top
                        )
                    )
                )
            )

    return app


def run_app(app, port=5000):
    app.run(port=port, debug=True)
    # http_server = WSGIServer(('', port), app)
    # webbrowser.open_new('http://localhost:' + str(port))
    # http_server.serve_forever()


def launch(model, classes=None, top=5, temp_folder='./tmp', input_folder='./', port=5000, html_base_dir=None):
    os.system('mkdir -p %s' % temp_folder)

    html_base_dir = html_base_dir if html_base_dir is not None else dirname(abspath(__file__))
    print('Starting webserver from:', html_base_dir)
    assert os.path.exists(os.path.join(html_base_dir, "quiverboard")), "Quiverboard must be a " \
                                                                       "subdirectory of {}".format(html_base_dir)
    assert os.path.exists(os.path.join(html_base_dir, "quiverboard", "dist")), "Dist must be a " \
                                                                               "subdirectory of quiverboard"
    assert os.path.exists(
        os.path.join(html_base_dir, "quiverboard", "dist", "index.html")), "Index.html missing"

    return run_app(
        get_app(model, classes, top, html_base_dir=html_base_dir,
                temp_folder=temp_folder, input_folder=input_folder),
        port
    )


def get_output_name(temp_folder, layer_name, input_path, z_idx):
    if os.pathsep in input_path or 'tmp' in input_path:
        input_path = os.path.basename(input_path)

    return temp_folder + '/' + layer_name + '_' + str(z_idx) + '_' + input_path + '.png'


def decode_predictions(preds, classes, top):
    if classes == 'regression':
        return [("", "Maize", preds)]

    if not classes:
        print("Warning! you didn't pass your own set of classes for the model therefore imagenet classes are used")
        return decode_imagenet_predictions(preds, top)

    if len(preds.shape) != 2 or preds.shape[1] != len(classes):
        raise ValueError('you need to provide same number of classes as model prediction output ' + \
                         'model returns %s predictions, while there are %s classes' % (
                             preds.shape[1], len(classes)))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [("", classes[i], pred[i]) for i in top_indices]
        results.append(result)

    return results

def get_evaluation_context_getter():
    if keras.backend.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.get_default_graph().as_default

    if keras.backend.backend() == 'theano':
        return contextmanager(lambda: (yield))
