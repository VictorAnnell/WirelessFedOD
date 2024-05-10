import random
from typing import Callable

import keras
import keras_cv
import numpy as np
import tensorflow as tf
from config import BATCH_SIZE


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
        self.preprocess_fn = None
        self.test_data = None
        self.train_data = train_data
        self.preprocessed_test_data = None
        self.preprocessed_train_data = None
        self.local_weights = None
        self.global_weights = None
        self.simulation_id = simulation_id
        self.test_split = 0.2
        self.round_num = 0
        self.deviation = None
        self.loss = None
        self.bit_rate = None

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

    def train(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.train_data is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            # Split train data into train/test with ratio 'test_split', giving test at least BATCH_SIZE elements
            train_size = tf.data.experimental.cardinality(self.train_data).numpy()
            test_size = int(train_size * self.test_split)
            test_size = max(test_size, BATCH_SIZE)
            self.test_data = self.train_data.take(test_size)
            self.train_data = self.train_data.skip(test_size)

        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        model = self.model_fn()
        model.set_weights(self.global_weights)

        if self.preprocessed_train_data is None:
            print("Preprocessing train data for", self)
            self.preprocessed_train_data = self.preprocess_fn(self.train_data)
        if self.preprocessed_test_data is None:
            print("Preprocessing test data for", self)
            self.preprocessed_test_data = self.preprocess_fn(self.test_data, validation_dataset=True)

        print(f"Training {self}")

        coco_metrics_callback = keras_cv.callbacks.PyCOCOCallback(
            self.preprocessed_test_data, bounding_box_format="xyxy", cache=False
        )
        result = model.fit(
            self.preprocessed_train_data,
            validation_data=self.preprocessed_test_data,
            initial_epoch=self.round_num * self.local_epochs,
            epochs=(self.round_num * self.local_epochs) + self.local_epochs,
            callbacks=[coco_metrics_callback] + self.callbacks,
            steps_per_epoch=self.steps_per_epoch,
        )
        # print(result.history)
        self.local_weights = model.get_weights()

        # Set loss
        self.loss = result.history["loss"][-1]

        # Set deviation
        flt_global_weights = np.concatenate(np.asanyarray(self.global_weights, dtype=object), axis=None)
        flt_local_weights = np.concatenate(np.asanyarray(self.local_weights, dtype=object), axis=None)
        self.deviation = np.linalg.norm(flt_global_weights - flt_local_weights)
        # print(
        #     f"{self} deviation: {self.deviation}, local weights sum: {np.sum(flt_local_weights)}, global weights sum: {np.sum(flt_global_weights)}"
        # )

        # Clear preproccessed data to save memory
        self.preprocessed_test_data = None
        self.preprocessed_train_data = None
