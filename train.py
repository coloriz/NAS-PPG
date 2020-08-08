from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Dropout, Flatten, LeakyReLU, LSTM, MaxPooling2D, Softmax, TimeDistributed
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils import get_latest_model, bpm_absolute_error

timesteps = 6
train_batch_size = 1
validation_batch_size = 1024
epochs = 10000
lr = 0.0001
model_name = 'baseline'


def build_model():
    two_power_spectra = Input(shape=(timesteps, 2, 222, 1), name='Two_Power_Spectra')
    acc_intensity = Input(shape=(timesteps, 1), name='Acc_intensity')
    x = TimeDistributed(Conv2D(32, (2, 37), 4, 'same', name='Conv1'), name='TD_Conv1')(two_power_spectra)
    x = LeakyReLU(name='Leaky_ReLU1')(x)
    x = TimeDistributed(MaxPooling2D((1, 2), name='Maxpooling1'), name='TD_Maxpooling1')(x)
    x = TimeDistributed(Dropout(0.3, name='Dropout1'), name='TD_Dropout1')(x)

    x = TimeDistributed(Conv2D(64, (1, 5), (1, 1), 'same', name='Conv2'), name='TD_Conv2')(x)
    x = LeakyReLU(name='Leaky_ReLU2')(x)
    x = TimeDistributed(MaxPooling2D((1, 2), name='Maxpooling2'), name='TD_Maxpooling2')(x)
    x = TimeDistributed(Dropout(0.3, name='Dropout2'), name='TD_Dropout2')(x)

    x = TimeDistributed(Flatten(name='Flatten'), name='TD_Flatten')(x)
    x = Dense(512, name='FC1')(x)
    x = LeakyReLU(name='Leaky_ReLU3')(x)

    x = Concatenate(name='Concatenate')([x, acc_intensity])
    x = LSTM(512, dropout=0.3, recurrent_dropout=0.2, return_sequences=True, name='LSTM1')(x)
    x = LSTM(222, dropout=0.3, recurrent_dropout=0.2, name='LSTM2')(x)

    x = Dense(222, name='FC2')(x)
    y = Softmax(name='Softmax')(x)

    return Model(inputs=[two_power_spectra, acc_intensity], outputs=y, name=model_name)


def train():
    # Load training data onto memory
    subjects_train = [np.load(fr'preprocessed/ISPC2015/subject_{i:02d}.npz') for i in range(1, 24)] + \
                     [np.load(fr'preprocessed/BAMI-1/subject_{i:02d}.npz') for i in range(1, 26)]
    # Load validation data onto memory
    subjects_validation = [np.load(fr'preprocessed/BAMI-2/subject_{i:02d}.npz') for i in range(1, 24)]

    def generate_train_sample(timesteps: int):
        for s in subjects_train:
            PS, PA, Ia, y_true = s['PS'], s['PA'], s['Ia'], s['y_true']
            for i in range(0, len(PS) - timesteps + 1):
                PS_i_series = PS[i:i + timesteps, ..., np.newaxis]
                PA_i_series = PA[i:i + timesteps, ..., np.newaxis]
                Ia_i_series = Ia[i:i + timesteps, ..., np.newaxis]
                y = y_true[i + timesteps - 1]

                two_ps = np.stack((PS_i_series, PA_i_series), axis=1)

                yield {'Two_Power_Spectra': two_ps, 'Acc_intensity': Ia_i_series}, y

    def generate_validation_sample(timesteps: int):
        for s in subjects_validation:
            PS, PA, Ia, y_true = s['PS'], s['PA'], s['Ia'], s['y_true']
            for i in range(0, len(PS) - timesteps + 1):
                PS_i_series = PS[i:i + timesteps, ..., np.newaxis]
                PA_i_series = PA[i:i + timesteps, ..., np.newaxis]
                Ia_i_series = Ia[i:i + timesteps, ..., np.newaxis]
                y = y_true[i + timesteps - 1]

                two_ps = np.stack((PS_i_series, PA_i_series), axis=1)

                yield {'Two_Power_Spectra': two_ps, 'Acc_intensity': Ia_i_series}, y

    dataset_train = tf.data.Dataset.from_generator(
        generate_train_sample, args=[timesteps],
        output_types=({'Two_Power_Spectra': tf.float32, 'Acc_intensity': tf.float32}, tf.float32),
        output_shapes=({'Two_Power_Spectra': (timesteps, 2, 222, 1), 'Acc_intensity': (timesteps, 1)}, (222,))
    ).shuffle(2 ** 14).batch(train_batch_size)

    dataset_validation = tf.data.Dataset.from_generator(
        generate_validation_sample, args=[timesteps],
        output_types=({'Two_Power_Spectra': tf.float32, 'Acc_intensity': tf.float32}, tf.float32),
        output_shapes=({'Two_Power_Spectra': (timesteps, 2, 222, 1), 'Acc_intensity': (timesteps, 1)}, (222,))
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
