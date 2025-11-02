from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf


class BrainTumorDataset:
    """Handle brain tumor MRI dataset loading and preprocessing"""

    def __init__(self, data_dir: Path, image_size: Tuple[int, int] = (150, 150)):
        self.data_dir = data_dir
        self.image_size = image_size

    def load_data(self, validation_split: float = 0.2):
        """Load and split dataset"""

        # Use TensorFlow's image_dataset_from_directory
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="training",
            seed=42,
            image_size=self.image_size,
            batch_size=32,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=42,
            image_size=self.image_size,
            batch_size=32,
        )

        return train_ds, val_ds

    def preprocess(self, dataset):
        """Normalize and augment data"""
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

        dataset = dataset.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
