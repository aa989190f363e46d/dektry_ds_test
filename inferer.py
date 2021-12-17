from functools import partial
import os

import tensorflow as tf
from tensorflow import keras

import utils


if __name__ == '__main__':

    DATA_DIR = os.environ.get('DATA_DIR', '.')
    img_file = input('Input test string: ')

    img_path = f'{DATA_DIR}/{img_file}.png'

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
    img = encode_image(tf.io.read_file(img_path))

    prediction = prediction_model.predict(tf.expand_dims(img, axis=0))
    pred_texts = utils.decode_batch_predictions(
        num_to_label,
        max_length,
        prediction,
    )

    print(pred_texts)
