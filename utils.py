import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Mapping characters to integers
def char_to_num(characters=None):
    return layers.StringLookup(
        vocabulary=list(characters),
        mask_token=None,
        )


# Mapping integers back to original characters
def num_to_char(vocab):
    return layers.StringLookup(
        vocabulary=vocab,
        mask_token=None,
        invert=True,
    )


def label_to_num(char_to_num_fn, label):
    return char_to_num_fn(tf.strings.unicode_split(
        label, input_encoding='UTF-8'),
    )


def num_to_label(num_to_char_fn, num):
    chars = num_to_char_fn(num)
    tensor = tf.strings.reduce_join(chars)
    return tensor.numpy().decode('utf-8')


def encode_single_sample(encode_image_fn, label_to_num_fn, img_path, label):
    # Read image
    img = tf.io.read_file(img_path)
    # Return a dict as our model is expecting two inputs
    return {
        'image': encode_image_fn(img),
        'label': label_to_num_fn(label),
        }


def encode_image(img_width, img_height, img):
    # Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    return img


def decode_batch_predictions(num_to_label_fn, max_length, pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    encoded_results = keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True)[0][0][:, :max_length]
    # Iterate over the results and get back the text
    return list(map(num_to_label_fn, encoded_results))


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(labels)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = (
        images[indices[:train_samples]],
        labels[indices[:train_samples]],
    )
    x_valid, y_valid = (
        images[indices[train_samples:]],
        labels[indices[train_samples:]],
    )
    return x_train, x_valid, y_train, y_valid


def get_dataset(encode_single_sample_fn, batch_sz, xs, ys):
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    return (
        dataset.map(
            encode_single_sample_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            ).batch(batch_sz).prefetch(buffer_size=tf.data.AUTOTUNE)
    )


def load_model():

    model_path = os.environ.get('MODEL_PATH', 'prediction_model')
    vocab_path = os.environ.get('VOCAB_PATH', 'vocab.txt')

    model = keras.models.load_model(model_path)

    with open(vocab_path, 'r') as vocab_file:
        vocab = list(vocab_file.readline().strip())

    return model, vocab


def save_model(model, vocab):

    model_path = os.environ.get('MODEL_PATH', 'prediction_model')
    vocab_path = os.environ.get('VOCAB_PATH', 'vocab.txt')

    model.save(model_path)

    # # Convert the model
    # converter = tf.lite.TFLiteConverter.from_saved_model('prediction_model')
    # tflite_model = converter.convert()

    # # Save the model.
    # with open('model.tflite', 'wb') as f:
    #     f.write(tflite_model)

    with open(vocab_path, 'w') as vocab_file:
        vocab_file.write(str.join('', vocab))
