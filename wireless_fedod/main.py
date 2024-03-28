import collections

import keras
import numpy as np
import tensorflow as tf
import keras_cv
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from dataset import load_zod_federated, OBJECT_CLASSES, class_mapping

np.random.seed(0)

BATCH_SIZE = 4

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )
    plt.show()

def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )
    plt.show()

class VisualizeDetections(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        visualize_detections(
            self.model, bounding_box_format="xywh", dataset=visualization_ds
        )

class Car:
    def __init__(self, id, model_fn, train_data, test_data):
        self.id = id
        self.local_epochs = 1
        self.location = [0, 0]
        self.classification_loss="binary_crossentropy"
        self.box_loss="ciou"
        self.model_fn = model_fn
        self.optimizer_fn = None
        self.preprocess_fn = None
        self.test_data = test_data
        self.train_data = train_data
        self.preprocessed_test_data = None
        self.preprocessed_train_data = None
        self.weights = None
        self.metrics = None

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

        # self.metrics = keras_cv.callbacks.PyCOCOCallback(self.preprocess_fn(self.test_data), bounding_box_format="xyxy")
        model = self.model_fn()
        model.set_weights(self.weights)
        # model.compile(optimizer=self.optimizer, classification_loss=self.classification_loss, box_loss=self.box_loss, metrics=self.metrics)
        model.compile(optimizer=self.optimizer_fn(), classification_loss=self.classification_loss, box_loss=self.box_loss)
        if self.preprocessed_train_data is None:
            print("Preprocessing train data for", self)
            self.preprocessed_train_data = self.preprocess_fn(self.train_data)
        # if self.preprocessed_test_data is None:
        #     self.preprocessed_test_data = self.preprocess_fn(self.test_data)
        print(f"Training {self}")
        model.fit(
            self.preprocessed_train_data,
            epochs=self.local_epochs,
            # validation_data=self.preprocessed_test_data,
            # callbacks=[self.metrics],
        )
        # model(preprocessed_train_data)
        self.weights = model.get_weights()

def default_preprocess(self, dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element["pixels"], [-1, 784]),
            y=tf.reshape(element["label"], [-1, 1]),
        )

    return (
        dataset.repeat(self.num_epochs)
        .shuffle(self.shuffle_buffer, seed=1)
        .batch(self.batch_size)
        .map(batch_format_fn)
        .prefetch(self.prefetch_buffer)
    )

def default_agent_selection(client_ids, num_clients):
    clients = np.random.choice(client_ids, num_clients//2, replace=False)
    if len(clients) == 0:
        clients = client_ids
    return clients


class WirelessFedODSimulator:
    def __init__(self, num_clients=2):
        self.train_data_list = None
        self.test_data = None
        self.federated_train_data = None
        self.model_fn = None
        self.agent_selection_fn = default_agent_selection
        self.preprocess_fn = default_preprocess
        self.num_epochs = 5
        self.batch_size = 20
        self.shuffle_buffer = 100
        self.prefetch_buffer = 10
        self.round_num = 0
        self.metrics = None
        self.cars = []
        self.global_weights = None
        self.num_clients = num_clients

        self.training_process = None
        self.train_state = None
        self.train_result = None
        self.sample_clients = None
        self.client_optimizer_fn = lambda: keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, global_clipnorm=10.0)
        self.server_optimizer_fn = lambda: keras.optimizers.SGD(learning_rate=1.0)

    # TODO: Handle the dataset in a more generic way
    def initialize_cars(self):
        for i in range(self.num_clients):
            car = Car(i, self.model_fn, self.train_data_list[i], self.test_data)
            car.optimizer_fn = self.client_optimizer_fn
            car.preprocess_fn = self.preprocess_fn
            self.cars.append(car)

    # @property
    # def num_clients(self):
    #     return len(self.cars)

    def set_hyperparameters(
        self,
        num_clients=10,
        num_epochs=5,
        batch_size=20,
        shuffle_buffer=100,
        prefetch_buffer=10,
    ):
        self.num_clients = num_clients
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer

    def initialize(self):
        self.round_num = 0
        self.global_weights = self.model_fn().get_weights()
        self.initialize_cars()

    def run_epoch(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.train_data_list is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        if self.global_weights is None or self.cars == []:
            self.initialize()

        cars_this_round = self.agent_selection_fn(self.cars, self.num_clients)
        if len(cars_this_round) == 0:
            raise ValueError("No clients selected.")

        # print(f"Selected clients: {cars_this_round}")
        print("Selected clients:", end=" ")
        print(", ".join(str(car) for car in cars_this_round))

        for car in tqdm(cars_this_round, desc="Training cars"):
            car.weights = self.global_weights
            car.train()

        # Update global weights
        # self.global_weights = np.mean([car.weights for car in cars_this_round], axis=0)
        self.round_num += 1

    def make_federated_data(self, client_data, client_ids):
        return [
            self.preprocess_fn(client_data.create_tf_dataset_for_client(x))
            for x in client_ids
        ]

    # def run_epoch(self):
    #     if self.train_state is None or self.round_num is None:
    #         self.initialize()
    #     self.round_num += 1
    #     self.sample_clients = self.agent_selection_fn(self.train_data, self.num_clients)
    #     self.federated_train_data = self.make_federated_data(
    #         self.train_data, self.sample_clients
    #     )
    #     self.train_result = self.training_process.next(
    #         self.train_state, self.federated_train_data
    #     )
    #     self.train_state = self.train_result.state
    #     self.metrics = self.train_result.metrics["client_work"]["train"]
    #     # Format metrics to be one line
    #     print("round {:2d}, ".format(self.round_num), end="")
    #     print(f"num_clients: {len(self.sample_clients)}", end=", ")
    #     self.print_metrics(self.train_result)

    # def initialize(self):
    #     self.round_num = 0
    #     self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
    #         self.model_fn,
    #         client_optimizer_fn=self.client_optimizer_fn,
    #         server_optimizer_fn=self.server_optimizer_fn,
    #     )
    #
    #     self.train_state = self.training_process.initialize()

    def print_metrics(self, train_result):
        for name, value in train_result.metrics["client_work"]["train"].items():
            print(f"{name}: {value:.4f}", end=", ")
        print()




# Set preprocess_fn
def preprocess_fn(dataset):
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        # image = keras.utils.load_img(image_path)
        return image

    def format_element_fn(image_path, classes, bboxes):
        # Create a dictionary with the image and bounding boxes as required by KerasCV
        image = load_image(image_path)
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": bboxes,
        }
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

    # def dict_to_tuple_fn(element):
    #     return element["images"], element["bounding_boxes"]

    def dict_to_tuple_fn(inputs):
        return inputs["images"], keras_cv.bounding_box.to_dense(
            inputs["bounding_boxes"], max_boxes=32
        )

    augmenters = [
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
        ),
    ]

    inference_resizing = keras_cv.layers.Resizing(
        640, 640, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
    )

    dataset = dataset.map(format_element_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BATCH_SIZE * 4, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.ragged_batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    def create_augmenter_fn(augmenters):
        def augmenter_fn(inputs):
            for augmenter in augmenters:
                inputs = augmenter(inputs)
            return inputs

        return augmenter_fn
    augmenter_fn = create_augmenter_fn(augmenters)
    dataset = dataset.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(dict_to_tuple_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat(2) # Repeat for more epochs
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


simulator = WirelessFedODSimulator()
simulator.preprocess_fn = preprocess_fn

# Load ZOD dataset
zod_train, zod_test = load_zod_federated(num_clients=simulator.num_clients)
# example_dataset = zod_train.create_tf_dataset_for_client(zod_train.client_ids[0])
# preprocessed_example_dataset = simulator.preprocess_fn(example_dataset)


# Create model_fn
def model_fn():
    return keras_cv.models.YOLOV8Detector.from_preset(
        "resnet50_imagenet",
        # "yolo_v8_m_pascalvoc",
        # For more info on supported bounding box formats, visit
        # https://keras.io/api/keras_cv/bounding_box/
        bounding_box_format="xyxy",
        num_classes=len(OBJECT_CLASSES),
    )
    # return keras_cv.models.YOLOV8Detector.from_preset(
    #     "yolo_v8_m_pascalvoc", bounding_box_format="xyxy"
    # )


# Set dataset, model_fn, and agent_selection_fn
simulator.train_data_list = zod_train
simulator.test_data = zod_test
simulator.model_fn = model_fn
# simulator.agent_selection_fn = lambda x, y: np.random.choice(x.client_ids, y, replace=False)
# Set optimizers
# simulator.client_optimizer_fn = lambda: keras.optimizers.Adam(learning_rate=0.001)
# simulator.server_optimizer_fn = lambda: keras.optimizers.Adam(learning_rate=0.1)
# Run training
simulator.run_epoch()

# if __name__ == '__main__':
def main():
    # Example Usage
    # Create simulator
    simulator = WirelessFedODSimulator()
    # Create train/test dataset
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        cache_dir="./cache"
    )
    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0]
    )
    preprocessed_example_dataset = simulator.preprocess_fn(example_dataset)

    # Create model_fn
    def create_keras_model():
        return keras.models.Sequential(
            [
                keras.layers.InputLayer(input_shape=(784,)),
                keras.layers.Dense(10, kernel_initializer="zeros"),
                keras.layers.Softmax(),
            ]
        )

    def model_fn():
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = create_keras_model()
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=preprocessed_example_dataset.element_spec,
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    # Set dataset, model_fn, and agent_selection_fn
    simulator.set_dataset(emnist_train, emnist_test)
    simulator.set_model_fn(model_fn)
    simulator.set_agent_selection_fn(
        lambda x, y: np.random.choice(x.client_ids, y, replace=False)
    )
    # Run training
    # while simulator.metrics is None or simulator.metrics['loss'] > 1:
    #     simulator.run_epoch()
    simulator.run_epoch()
    # Plot metrics
    # plt.plot(simulator.metrics['loss'])
    # plt.plot(simulator.metrics['sparse_categorical_accuracy'])
    # plt.show()
