from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Conv1D, Flatten, LeakyReLU, LSTM, Softmax, TimeDistributed
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2

from utils import get_latest_model, bpm_absolute_error

timesteps = 6
train_batch_size = 1
validation_batch_size = 1024
epochs = 10000
lr = 0.0001
model_name = 'v2-rc1'


def build_model():
    two_power_spectra = Input(shape=(timesteps, 222, 2), name='Two_Power_Spectra')
    x = TimeDistributed(Conv1D(16, 11, 3, 'same', kernel_regularizer=L2(), name='Conv1'), name='TD_Conv1')(two_power_spectra)
    x = BatchNormalization(name='BN1')(x)
    x = LeakyReLU(name='Leaky_ReLU1')(x)

    x = TimeDistributed(Flatten(name='Flatten'), name='TD_Flatten')(x)

    x = LSTM(222, kernel_regularizer=L2(), recurrent_regularizer=L2(), name='LSTM1')(x)
    y = Softmax(name='Softmax')(x)

    return Model(inputs=two_power_spectra, outputs=y, name=model_name)


def train():
    # Load training data onto memory
    subjects_train = [np.load(fr'preprocessed/ISPC2015/subject_{i:02d}.npz') for i in range(1, 24)] + \
                     [np.load(fr'preprocessed/BAMI-1/subject_{i:02d}.npz') for i in range(1, 26)]
    # Load validation data onto memory
    subjects_validation = [np.load(fr'preprocessed/BAMI-2/subject_{i:02d}.npz') for i in range(1, 24)]

    def generate_train_sample(timesteps: int):
        for s in subjects_train:
            PS, PA, y_true = s['PS'], s['PA'], s['y_true']
            for i in range(0, len(PS) - timesteps + 1):
                PS_i_series = PS[i:i + timesteps]
                PA_i_series = PA[i:i + timesteps]
                y = y_true[i + timesteps - 1]

                two_ps = np.stack((PS_i_series, PA_i_series), axis=-1)

                yield two_ps, y

    def generate_validation_sample(timesteps: int):
        for s in subjects_validation:
            PS, PA, y_true = s['PS'], s['PA'], s['y_true']
            for i in range(0, len(PS) - timesteps + 1):
                PS_i_series = PS[i:i + timesteps]
                PA_i_series = PA[i:i + timesteps]
                y = y_true[i + timesteps - 1]

                two_ps = np.stack((PS_i_series, PA_i_series), axis=-1)

                yield two_ps, y

    dataset_train = tf.data.Dataset.from_generator(
        generate_train_sample, args=[timesteps],
        output_types=(tf.float32, tf.float32),
        output_shapes=((timesteps, 222, 2), (222,))
    ).shuffle(2 ** 14).batch(train_batch_size)

    dataset_validation = tf.data.Dataset.from_generator(
        generate_validation_sample, args=[timesteps],
        output_types=(tf.float32, tf.float32),
        output_shapes=((timesteps, 222, 2), (222,))
    ).batch(validation_batch_size)

    # Check if there is a model
    latest_model = get_latest_model(Path(f'models/{model_name}'))
    if latest_model:
        model = load_model(str(latest_model), custom_objects={
            'bpm_absolute_error': bpm_absolute_error,
        })
        initial_epoch = int(latest_model.name.split('-')[1])
        print(f'Found existing model. Start training from here... (epoch: {initial_epoch}')
    else:
        model = build_model()
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=CategoricalCrossentropy(from_logits=False),
            metrics=[CategoricalAccuracy(), bpm_absolute_error],
        )
        initial_epoch = 0
        print(f'Start training from scratch...')
    model.summary(110)

    callbacks = [
        TensorBoard(log_dir=f'logs/{model_name}/'),
        ModelCheckpoint(
            f'models/{model_name}/model-{{epoch:04d}}-{{val_loss:.4f}}-{{val_bpm_absolute_error:.4f}}',
            monitor='val_bpm_absolute_error',
            save_best_only=True,
            mode='min'),
    ]
    model.fit(
        dataset_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=dataset_validation,
        initial_epoch=initial_epoch)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    train()
