import datetime

import gc
import keras
import keras_cv
import numpy as np
import tensorflow as tf

from wireless_fedod.base_station import BaseStation
from wireless_fedod.car import Car
from wireless_fedod.config import (
    CLASS_MAPPING,
    IMPORTANCE_FN,
    LOCAL_EPOCHS,
    MIXED_PRECISION,
    MODEL_FN,
    NUM_CLIENTS,
    SIMULATION_ID,
    STEPS_PER_LOCAL_EPOCH,
)
from wireless_fedod.dataset import noniid_split_dataset, preprocess_fn
from wireless_fedod.utils import EvaluateCOCOMetricsCallback, fedavg_aggregate, visualize_dataset, visualize_detection


class WirelessFedODSimulator:
    def __init__(self, num_clients=NUM_CLIENTS, simulation_id=SIMULATION_ID):
        self.simulation_id = (
            simulation_id + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            if simulation_id
            else datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
        self.train_data = None
        self.test_data = None
        self.model_fn = MODEL_FN
        self.metrics = {}
        self.importance_fn = IMPORTANCE_FN
        self.preprocess_fn = preprocess_fn
        self.local_epochs = LOCAL_EPOCHS
        self.steps_per_local_epoch = STEPS_PER_LOCAL_EPOCH
        self.round_num = 0
        self.cars = []
        self.global_weights = None
        self.num_clients = num_clients
        self.callbacks = []
        self.base_stations = []
        self.base_station = BaseStation(0, (0, 0))

    def __str__(self):
        return f"WirelessFedODSimulator {self.simulation_id}"

    def initialize_cars(self):
        if self.train_data is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        train_data_splits = noniid_split_dataset(self.train_data, self.num_clients)
        print("Initializing cars")
        for i in range(self.num_clients):
            car = Car(
                i,
                self.model_fn,
                train_data_splits[i],
                self.simulation_id,
                local_epochs=self.local_epochs,
                steps_per_epoch=self.steps_per_local_epoch,
            )
            car.preprocess_fn = self.preprocess_fn
            car.bit_rate = self.base_station.get_car_bit_rate(car)
            self.cars.append(car)

    def initialize(self):
        self.callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=f"logs/{self.simulation_id}/global",
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=True,
            ),
        )
        self._file_writer = tf.summary.create_file_writer(f"logs/{self.simulation_id}/global/eval")
        self.round_num = 0
        self.global_weights = self.model_fn().get_weights()
        self.initialize_cars()

    def run_round(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.train_data is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        if self.global_weights is None or self.cars == []:
            self.initialize()

        # Get cars with highest importance scores
        self.cars_this_round = sorted(self.cars, key=lambda x: x.importance, reverse=True)[: max(len(self.cars) // 2, 1)]
        if len(self.cars_this_round) == 0:
            raise ValueError("No clients selected.")

        print("Selected clients:", end=" ")
        print(", ".join(str(car) for car in self.cars_this_round))

        for car in self.cars_this_round:
            car.global_weights = self.global_weights
            car.train()
            try: # Keras 3
                keras.backend.clear_session(free_memory=True)
            except TypeError: # Keras 2
                keras.backend.clear_session()
            # if MIXED_PRECISION:
            #     print('Mixed precision enabled')
            #     keras.mixed_precision.set_global_policy("mixed_float16")
            # tf.compat.v1.reset_default_graph()
            # gc.collect()
            # TODO: set seed again?

        # Update global weights with the scaled average of the local weights
        total_samples = sum(len(car.train_data) for car in self.cars_this_round)
        weighted_weights = [
            [layer * len(car.train_data) / total_samples for layer in car.local_weights] for car in self.cars_this_round
        ]
        self.global_weights = [np.average(weights, axis=0) for weights in zip(*weighted_weights)]

        # reference implementation
        self.global_weights = fedavg_aggregate(self.cars, self.cars_this_round)

        self.round_num += 1
        # Communicate round number to cars
        for car in self.cars:
            car.round_num = self.round_num
            car.importance = self.importance_fn(self, car)

        self.evaluate()

    def evaluate(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")
        if self.global_weights is None:
            raise ValueError("Global weights are not set, run a round first.")

        print("Evaluating global model")
        model = self.model_fn()
        model.set_weights(self.global_weights)
        preprocessed_test_data = self.preprocess_fn(self.test_data, validation_dataset=True)
        result = model.evaluate(preprocessed_test_data, callbacks=[
            EvaluateCOCOMetricsCallback(preprocessed_test_data, "round_model.h5")
        ] + self.callbacks)
        # Store metrics
        if isinstance(result, list):
            self.metrics = dict(zip(model.metrics_names, result))
        else:
            self.metrics = {"loss": result}
        self.print_metrics(model, result)

    def visualize_detection(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")
        if self.global_weights is None:
            print("Global weights are not set, using initial weights")
            self.initialize()

        print("Visualizing detection")
        model = self.model_fn()
        model.set_weights(self.global_weights)
        visualize_detection(model, self.test_data, self.preprocess_fn, class_mapping=CLASS_MAPPING)

    def visualize_dataset(self, type="test"):
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        if type == "test":
            if self.test_data is None:
                raise ValueError("Test data is not set.")
            dataset = self.test_data
        elif type == "train":
            if self.train_data is None:
                raise ValueError("Train data is not set.")
            dataset = self.train_data
        else:
            raise ValueError("Invalid dataset type. Should be 'train' or 'test'")

        print(f"Visualizing {type} dataset")
        visualize_dataset(dataset, self.preprocess_fn, class_mapping=CLASS_MAPPING)

    def print_metrics(self, model, result):
        print()
        print("Round Metrics:")
        with self._file_writer.as_default(step=self.round_num):
            if isinstance(result, list):
                for name, value in zip(model.metrics_names, result):
                    print(f"{name}: {value:.4f}", end=", ")
                    tf.summary.scalar(f"round_{name}", value)
            else:
                print(f"loss: {result:.4f}", end=", ")
                tf.summary.scalar("round_loss", result)
        print(end="\n\n")
