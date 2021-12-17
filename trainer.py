import numpy as np
import os

from functools import partial, reduce
from itertools import chain, starmap
from matplotlib import pyplot as plt
from operator import eq
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utils


def build_model(img_width, img_height, vocab_len):

    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1),
        name='image',
        dtype='float32',
    )
    labels = layers.Input(name='label', shape=(None,), dtype='float32')

    new_shape = ((img_width // 4), (img_height // 4) * 64)

    seq_mod = reduce(
        lambda lower, upper: upper(lower),
        # First conv block
        (
            layers.Conv2D(
                32, (3, 3),
                activation='relu',
                kernel_initializer='he_normal',
                padding='same',
                name='Conv1',
            ),
            layers.MaxPooling2D((2, 2), name='pool1'),
            # Second conv block
            layers.Conv2D(
                64, (3, 3),
                activation='relu',
                kernel_initializer='he_normal',
                padding='same',
                name='Conv2',
            ),
            layers.MaxPooling2D((2, 2), name='pool2'),
            # We have used two max pool with pool size and strides 2.
            # Hence, downsampled feature maps are 4x smaller. The number of
            # filters in the last layer is 64. Reshape accordingly before
            # passing the output to the RNN part of the model
            layers.Reshape(target_shape=new_shape, name='reshape'),
            layers.Dense(64, activation='relu', name='dense1'),
            layers.Dropout(0.2),
            # RNNs
            layers.Bidirectional(
                layers.LSTM(
                    128,
                    return_sequences=True,
                    dropout=0.25,
                ),
            ),
            layers.Bidirectional(
                layers.LSTM(
                    64,
                    return_sequences=True,
                    dropout=0.25,
                ),
            ),

            # Output layer
            layers.Dense(
                vocab_len + 1,
                activation='softmax',
                name='dense2',
            ),
        ),
        input_img,
    )

    class CTCLayer(layers.Layer):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.loss_fn = keras.backend.ctc_batch_cost

        def call(self, y_true, y_pred):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
            input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
            label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')
            loss = self.loss_fn(
                y_true, y_pred,
                input_length * tf.ones(shape=(batch_len, 1), dtype='int64'),
                label_length * tf.ones(shape=(batch_len, 1), dtype='int64'),
            )
            self.add_loss(loss)

            # At test time, just return the computed predictions
            return y_pred

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name='ctc_loss')(labels, seq_mod)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels],
        outputs=output,
        name='dektry_test_ocr_model_v1',
    )

    # Optimizer
    opt = keras.optimizers.Adam()

    # Compile the model and return
    model.compile(optimizer=opt, metrics=[])

    return model


def eval_model(
        model,
        num_to_label_fn,
        max_length,
        train_dataset,
        validation_dataset,
        ):
    val_preds = []
    for batch in chain(train_dataset, validation_dataset):

        batch_images = batch['image']
        labels = map(num_to_label_fn, batch['label'])

        preds = model.predict(batch_images)
        pred_texts = utils.decode_batch_predictions(
            num_to_label_fn,
            max_length,
            preds,
        )

        val_preds.extend(zip(pred_texts, labels))

    with open('eval_results.txt', 'w') as f:
        for pred, label in val_preds:
            f.write(f'{pred}\t{label}\t{"*" if pred==label else ""}\n')

    print('Prec:', sum(starmap(eq, val_preds)) / len(val_preds))


def plot_batch():
    #  Let's check results on some validation samples
    cols = 4
    rows = BATCH_SIZE // cols
    for batch in validation_dataset.take(1):

        batch_images = batch['image']

        preds = prediction_model.predict(batch_images)
        pred_texts = utils.decode_batch_predictions(preds)

        _, ax = plt.subplots(cols, rows, figsize=(10, 5))
        for i, (b_img, label) in enumerate(zip(batch_images, pred_texts)):
            img = (b_img[:, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f'Prediction: {label}'
            col, row = i // cols, i % cols
            subplot = ax[row, col]
            subplot.imshow(img, cmap='gray')
            subplot.set_title(title)
            subplot.axis('off')

        plt.show()


if __name__ == '__main__':

    DATA_DIR = os.environ.get('DATA_DIR', './tagged')
    # Path to the data directory
    data_dir_path = Path(DATA_DIR)

    # Batch size for training and validation
    BATCH_SIZE = int(os.environ.get('BATCH', 16))
    EPOCHS = int(os.environ.get('EPOCHS', 100))

    # Get list of all the images
    images = list(map(str, data_dir_path.glob('*.png')))
    labels = [Path(img).stem for img in images]
    # Maximum length of any captcha in the dataset
    max_length = max(map(len, labels))
    characters = set(chain.from_iterable(map(list, labels)))

    # Mapping characters to integers
    char_to_num = utils.char_to_num(characters)
    vocab = char_to_num.get_vocabulary()
    num_to_char = utils.num_to_char(vocab)

    label_to_num = partial(utils.label_to_num, char_to_num)
    num_to_label = partial(utils.num_to_label, num_to_char)

    print('Number of labels found: ', len(labels))
    print('Number of unique characters: ', len(characters))
    print('Characters present: ', characters)

    # Desired image dimensions
    img_width = 214
    img_height = 60

    # Splitting data into training and validation sets
    x_train, x_valid, y_train, y_valid = utils.split_data(
        np.array(images),
        np.array(labels),
    )

    encode_image = partial(utils.encode_image, img_width, img_height)
    encode_single_sample = partial(
        utils.encode_single_sample,
        encode_image,
        label_to_num,
    )
    get_dataset = partial(utils.get_dataset, encode_single_sample, BATCH_SIZE)
    train_dataset = get_dataset(x_train, y_train)
    validation_dataset = get_dataset(x_valid, y_valid)

    # Get the model
    model = build_model(img_width, img_height, len(vocab))
    model.summary()

    """
    ## Training
    """

    early_stopping_patience = 10
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name='image').input,
        model.get_layer(name='dense2').output,
    )

    utils.save_model(prediction_model, characters)

    prediction_model.summary()

    eval_model(
        prediction_model,
        num_to_label,
        max_length,
        train_dataset,
        validation_dataset,
    )
