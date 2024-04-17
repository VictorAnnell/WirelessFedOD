import keras
import keras_cv
import numpy as np
import tensorflow as tf
from dataset import OBJECT_CLASSES, load_zod_federated
from simulator import WirelessFedODSimulator

BATCH_SIZE = 4


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def format_element_fn(image_path, classes, bboxes):
    # Create a dictionary with the image and bounding boxes as required by KerasCV
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bboxes,
    }
    return {
        "images": tf.cast(image, tf.float32),
        "bounding_boxes": bounding_boxes,
    }

def dict_to_tuple_fn(inputs):
    return inputs["images"], keras_cv.bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )

# Set preprocess_fn
def preprocess_fn(dataset, validation_dataset=False):
    if validation_dataset:
        augmenters = [
            keras_cv.layers.Resizing(
                640, 640, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
            )
        ]
    else:
        augmenters = [
            keras_cv.layers.RandomFlip(
                mode="horizontal", bounding_box_format="xyxy"
            ),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640),
                scale_factor=(0.75, 1.3),
                bounding_box_format="xyxy",
            ),
        ]

    dataset = dataset.map(format_element_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BATCH_SIZE * 4)
    dataset = dataset.ragged_batch(BATCH_SIZE, drop_remainder=True)

    def create_augmenter_fn(augmenters):
        def augmenter_fn(inputs):
            for augmenter in augmenters:
                inputs = augmenter(inputs)
            return inputs

        return augmenter_fn

    augmenter_fn = create_augmenter_fn(augmenters)
    dataset = dataset.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(dict_to_tuple_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat(2)  # Repeat for more epochs TODO: set this correctly
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create model_fn
def model_fn():
    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_xs_backbone"
        ),
        fpn_depth=2,
    )
    base_lr = 0.005
    # including a global_clipnorm is extremely important in object detection tasks
    optimizer = keras.optimizers.SGD(
        learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
    )
    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
    )
    return model

if __name__ == "__main__":
    simulator = WirelessFedODSimulator()
    simulator.preprocess_fn = preprocess_fn

    # Load ZOD dataset
    zod_train, zod_test = load_zod_federated(
        num_clients=simulator.num_clients, version="full", upper_bound=99
    )

    # Set dataset, model_fn, and agent_selection_fn
    simulator.train_data_list = zod_train
    simulator.test_data = preprocess_fn(zod_test, validation_dataset=True)
    simulator.model_fn = model_fn
    simulator.run_round()
