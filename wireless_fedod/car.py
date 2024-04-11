import keras_cv
import random
import tensorflow as tf


class Car:

    def __init__(self, id, model_fn, train_data, test_data, simulation_id, local_epochs=5):
        self.id = id
        self.local_epochs = local_epochs
        self.location = (random.uniform(-100, 100), random.uniform(-100, 100))
        self.classification_loss = "binary_crossentropy"
        self.box_loss = "ciou"
        self.model_fn = model_fn
        self.optimizer_fn = None
        self.preprocess_fn = None
        self.test_data = test_data
        self.train_data = train_data
        self.preprocessed_test_data = None
        self.preprocessed_train_data = None
        self.weights = None
        self.metrics = None
        self.callbacks = []
        self.simulation_id = simulation_id

    def __str__(self):
        return f"Car {self.id}"

    def train(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.optimizer_fn is None:
            raise ValueError("Optimizer is not set.")
        # if self.metrics is None:
        #     raise ValueError("Metrics are not set.")
        if self.train_data is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
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

        # Callbacks
        # self.callbacks.append(keras_cv.callbacks.PyCOCOCallback(self.test_data, bounding_box_format="xyxy"))
        log_dir = f"logs/{self.simulation_id}/cars/{self.id}"
        self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, write_steps_per_second=True))

        print(f"Training {self}")


        model.fit(
            self.preprocessed_train_data,
            epochs=self.local_epochs,
            # validation_data=self.preprocessed_test_data,
            callbacks=self.callbacks,
            verbose=2,
        )
        # model(preprocessed_train_data)
        self.weights = model.get_weights()
