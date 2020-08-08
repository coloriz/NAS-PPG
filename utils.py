from pathlib import Path
from typing import Optional

import tensorflow as tf

from preprocess import processor_BAMI2


def get_latest_model(model_path: Path) -> Optional[Path]:
    if not model_path.is_dir():
        return
    visible_folders = filter(lambda d: not d.name.startswith('.'), model_path.iterdir())
    return max(visible_folders, key=lambda ckpt: ckpt.name.split('-')[1])


def bpm_absolute_error(y_true, y_pred):
    true_index = tf.math.argmax(y_true, axis=-1)
    pred_index = tf.math.argmax(y_pred, axis=-1)
    index_difference = tf.math.abs(true_index - pred_index)
    bps = tf.cast(index_difference, tf.float32) * processor_BAMI2.frequency_resolution
    return tf.math.reduce_mean(60 * bps, axis=-1)
