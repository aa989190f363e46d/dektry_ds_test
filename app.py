from flask import Flask, jsonify, request

import os
import base64

from functools import partial

import tensorflow as tf
from tensorflow import keras

import utils

max_length = 6

img_width = 214
img_height = 60

prediction_model, vocab = utils.load_model()
# Mapping characters to integers
char_to_num = utils.char_to_num(vocab)
vocab = char_to_num.get_vocabulary()
num_to_char = utils.num_to_char(vocab)
num_to_label = partial(utils.num_to_label, num_to_char)
encode_image = partial(utils.encode_image, img_width, img_height)

app = Flask(__name__)

print('Initialization done.')


# Define a route for the default URL, which loads the form
@app.route('/', methods=['POST'])
def inference():
    b64 = base64.b64decode(request.form['b64'])
    
    prediction = prediction_model.predict(
        tf.expand_dims(
            encode_image(b64), axis=0),
        )
    pred_texts = utils.decode_batch_predictions(
        num_to_label,
        max_length,
        prediction,
    )

    return jsonify({'label': pred_texts})

        
if __name__ == "__main__":
    app.run(host='127.0.0.1')
