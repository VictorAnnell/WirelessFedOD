import random
from typing import Callable

import keras
import keras_cv
import numpy as np
import tensorflow as tf

from wireless_fedod.config import BATCH_SIZE, RECREATE_MODEL
from wireless_fedod.utils import EvaluateCOCOMetricsCallback


class Car:
    def __init__(
        self,
        id,
        model_fn: Callable[[], keras.Model],
        train_data: tf.data.Dataset,
        simulation_id: str,
        local_epochs: int = 1,
        steps_per_epoch=None,
    ):
        self.id = id
        self.local_epochs = local_epochs
        self.steps_per_epoch = steps_per_epoch
        self.location = (random.uniform(-100, 100), random.uniform(-100, 100))
        self.model_fn = model_fn
        self.model = None
        self.preprocess_fn = None
        self.test_data = None
        self.train_data = train_data
        self.local_weights = None
        self.global_weights = None
        self.simulation_id = simulation_id
        self.test_split = 0.2
        self.round_num = 0
        self.deviation = 0.0
        self.loss = None
        self.MaP = 0.0
        self.bit_rate = 0.0
        self.importance = 0.0

        # Callbacks
        self.callbacks = []
        log_dir = f"logs/{self.simulation_id}/cars/{self.id}"
        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=True,
            )
        )

    def __str__(self):
        return f"Car {self.id}"

    def initialize(self):
        if self.train_data is None:
            raise ValueError("Training data is not set.")

        if self.test_data is None:
            # Split train data into train/test with ratio 'test_split', giving test at least BATCH_SIZE elements
            train_size = tf.data.experimental.cardinality(self.train_data).numpy()
            test_size = int(train_size * self.test_split)
            test_size = max(test_size, BATCH_SIZE)
            self.test_data = self.train_data.take(test_size)
            self.train_data = self.train_data.skip(test_size)
            train_size = tf.data.experimental.cardinality(self.train_data).numpy()
            test_size = tf.data.experimental.cardinality(self.test_data).numpy()
            print(f"Car {self.id} train data size: {train_size}, test data size: {test_size}")

        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        self.train_data = self.preprocess_fn(self.train_data)
        self.test_data = self.preprocess_fn(self.test_data, validation_dataset=True)

    def train(self):
        if self.model_fn is None and self.model is None:
            raise ValueError("Neither a model function or model is set.")
        if self.train_data is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            self.initialize()

        if self.model is None or RECREATE_MODEL:
            self.model = self.model_fn()
        self.model.set_weights(self.global_weights)

        print(f"Training {self}")

        result = self.model.fit(
            self.train_data,
            validation_data=self.test_data,
            initial_epoch=self.round_num * self.local_epochs,
            epochs=(self.round_num * self.local_epochs) + self.local_epochs,
            callbacks=[EvaluateCOCOMetricsCallback(self.preprocessed_test_data, f"car_{self.id}_model.h5")]
            + self.callbacks,
            steps_per_epoch=self.steps_per_epoch,
        )
        self.local_weights = self.model.get_weights()

        # Set loss
        self.loss = result.history["loss"][-1]

        # Set MaP
        self.MaP = result.history["MaP"][-1]

        # Set deviation
        flt_global_weights = np.concatenate(np.asanyarray(self.global_weights, dtype=object), axis=None)
        flt_local_weights = np.concatenate(np.asanyarray(self.local_weights, dtype=object), axis=None)
        self.deviation = np.linalg.norm(flt_global_weights - flt_local_weights)
        # print(
        #     f"{self} deviation: {self.deviation}, local weights sum: {np.sum(flt_local_weights)}, global weights sum: {np.sum(flt_global_weights)}"
        # )
