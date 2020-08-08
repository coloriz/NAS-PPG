from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from preprocess import processor_BAMI2
from utils import get_latest_model, bpm_absolute_error

timesteps = 6
model_name = 'baseline'
test_batch_size = 1024


def run_test():
    # Load test data onto memory
    subjects_test = [np.load(fr'preprocessed/BAMI-2/subject_{i:02d}.npz') for i in range(1, 24)]

    def generate_x_test(timesteps: int):
        for s in subjects_test:
            PS, PA, Ia = s['PS'], s['PA'], s['Ia']
            for i in range(0, len(PS) - timesteps + 1):
                PS_i_series = PS[i:i + timesteps, ..., np.newaxis]
                PA_i_series = PA[i:i + timesteps, ..., np.newaxis]
                Ia_i_series = Ia[i:i + timesteps, ..., np.newaxis]

                two_ps = np.stack((PS_i_series, PA_i_series), axis=1)

                yield {'Two_Power_Spectra': two_ps, 'Acc_intensity': Ia_i_series}

    def generate_bpm_true(timesteps: int):
        for s in subjects_test:
            bpm = s['bpm']
            for i in range(0, len(bpm) - timesteps + 1):
                yield bpm[i + timesteps - 1]

    dataset_x_test = tf.data.Dataset.from_generator(
        generate_x_test, args=[timesteps],
        output_types={'Two_Power_Spectra': tf.float32, 'Acc_intensity': tf.float32},
        output_shapes={'Two_Power_Spectra': (timesteps, 2, 222, 1), 'Acc_intensity': (timesteps, 1)},
    ).batch(test_batch_size)

    bpm_true = np.array(list(generate_bpm_true(timesteps)))

    latest_checkpoint = get_latest_model(Path(fr'models/{model_name}/'))
    model = load_model(str(latest_checkpoint), custom_objects={
        'bpm_absolute_error': bpm_absolute_error,
    })
    model.summary()

    y_pred = model.predict(dataset_x_test)
    bpm_pred = processor_BAMI2.frequency_bin_to_bpm(np.argmax(y_pred, axis=-1))

    print(f'AAE = {np.mean(np.abs(bpm_true - bpm_pred))}')


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
    run_test()
