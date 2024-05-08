import random

import keras_cv
import keras
import tensorflow as tf
from typing import Callable


class Car:
    def __init__(
        self, id, model_fn: Callable[[], keras.Model], train_data: tf.data.Dataset, simulation_id: str, local_epochs: int = 1, steps_per_epoch = None
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
        self.weights = None
        self.simulation_id = simulation_id
        self.test_split = 0.2
        self.round_num = 0

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
            # Split train data into train and test
            len_train = len(self.train_data)
            len_test = int(len_train * self.test_split)
            self.test_data = self.train_data.take(len_test)
            self.train_data = self.train_data.skip(len_test)

        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        model = self.model_fn()
        model.set_weights(self.weights)

        if self.preprocessed_train_data is None:
            print("Preprocessing train data for", self)
            self.preprocessed_train_data = self.preprocess_fn(self.train_data)
        if self.preprocessed_test_data is None:
            print("Preprocessing test data for", self)
            self.preprocessed_test_data = self.preprocess_fn(self.test_data)


        print(f"Training {self}")

        coco_metrics_callback = keras_cv.callbacks.PyCOCOCallback(
            self.preprocessed_test_data, bounding_box_format="xyxy"
        )
        result = model.fit(
            self.preprocessed_train_data,
            validation_data=self.preprocessed_test_data,
            initial_epoch=self.round_num * self.local_epochs,
            epochs=(self.round_num * self.local_epochs) + self.local_epochs,
            callbacks=[coco_metrics_callback] + self.callbacks,
            steps_per_epoch=self.steps_per_epoch,
        )
        print(result.history)
        self.weights = model.get_weights()

        self.preprocessed_test_data = None
        self.preprocessed_train_data = None
