import random

import keras_cv
import numpy as np
import tensorflow as tf
import zod.constants as constants
from tqdm.auto import tqdm
from zod import ZodFrames
from zod.anno.object import OBJECT_CLASSES
from zod.constants import AnnotationProject, Anonymization

random.seed(0)

class_mapping = dict(zip(range(len(OBJECT_CLASSES)), OBJECT_CLASSES))


def create_dataset(zod_frames, frame_ids, bounding_box_format="xyxy"):
    # Load training_frames inta a tensorfow dataset
    image_paths = []
    bbox = []
    class_ids = []
    for frame_id in frame_ids:
        frame_bboxs = []
        frame_classes = []
        frame_has_2d_bbox = False
        frame = zod_frames[frame_id]
        image_path = frame.info.get_key_camera_frame(Anonymization.BLUR).filepath
        annotations = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        for annotation in annotations:
            if annotation.box2d:
                # Only include frame in dataset if it has 2d bounding boxes
                frame_has_2d_bbox = True
                frame_bboxs.append(annotation.box2d.xyxy)
                frame_classes.append(annotation.superclass)
        if frame_has_2d_bbox:
            image_paths.append(image_path)
            bbox.append(frame_bboxs)
            # Convert classes to class_ids
            frame_class_ids = [
                list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
                for cls in frame_classes
            ]
            class_ids.append(frame_class_ids)
    bbox_tensor = tf.ragged.constant(bbox)
    # TODO: fix
    converted_bbox_tensor = keras_cv.bounding_box.convert_format(
        bbox_tensor, bounding_box_format, "xyxy"
    )
    classes_tensor = tf.ragged.constant(class_ids)
    image_paths_tensor = tf.ragged.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths_tensor, classes_tensor, converted_bbox_tensor)
    )
    return dataset


def get_random_sized_subset(input_list, client_id, num_clients, seed):
    random.seed(seed)
    # Generate random subset sizes
    total_elements = len(input_list)
    subset_sizes = [
        random.randint(1, total_elements // num_clients + 1) for _ in range(num_clients)
    ]
    # Adjust the last subset size if the sum exceeds the list length
    while sum(subset_sizes) > total_elements:
        subset_sizes[-1] -= 1
    # Allocate subsets based on these sizes
    subsets = []
    start_index = 0
    for size in subset_sizes:
        subsets.append(input_list[start_index : start_index + size])
        start_index += size
    # Check if client_id is valid
    if client_id < 0 or client_id >= num_clients:
        raise ValueError("Invalid client_id")
    return subsets[client_id]


def load_zod(version="mini", seed=0, bounding_box_format="xyxy", upper_bound=None):
    # NOTE! Set the path to dataset and choose a version
    dataset_root = "../datasets"
    version = version  # "mini" or "full"

    # initialize ZodFrames
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

    # # get default training and validation splits
    training_frames = zod_frames.get_split(constants.TRAIN)
    validation_frames = zod_frames.get_split(constants.VAL)

    if upper_bound:
        training_frames = {x for x in training_frames if int(x) <= upper_bound}
        validation_frames = {x for x in validation_frames if int(x) <= upper_bound}

    training_dataset = create_dataset(
        zod_frames, training_frames, bounding_box_format=bounding_box_format
    )
    validation_dataset = create_dataset(
        zod_frames, validation_frames, bounding_box_format=bounding_box_format
    )

    return training_dataset, validation_dataset

def noniid_split_dataset(dataset: 'tf.data.Dataset', num_splits: int, alpha: int = 1) -> list('tf.data.Dataset'):
    """
    Split a dataset into num_splits non-iid datasets.
    """
    if num_splits > len(dataset):
        raise ValueError("Number of splits cannot exceed the number of elements in dataset.")

    # Assign one element to each split to ensure that each split is non-empty
    base_dataset = dataset.take(num_splits)
    dataset = dataset.skip(num_splits)
    # Generate Dirichlet distribution proportions
    proportions = np.random.dirichlet(alpha * np.ones(num_splits))
    # Calculate number of elements per split
    num_elements = np.round(proportions * len(dataset)).astype(int)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=1000)
    # Distribute the dataset into num_splits
    dataset_splits = []
    for num in num_elements:
        split_base = base_dataset.take(1)
        base_dataset = base_dataset.skip(1)
        dataset_splits.append(split_base.concatenate(dataset.take(num)))
        dataset = dataset.skip(num)

    return dataset_splits


def load_zod_federated(
    num_clients=5, version="mini", seed=0, bounding_box_format="xyxy", upper_bound=None
):
    # NOTE! Set the path to dataset and choose a version
    dataset_root = "../datasets"
    version = version  # "mini" or "full"

    # initialize ZodFrames
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

    # # get default training and validation splits
    training_frames = zod_frames.get_split(constants.TRAIN)
    validation_frames = zod_frames.get_split(constants.VAL)

    if upper_bound:
        training_frames = {x for x in training_frames if int(x) <= upper_bound}
        validation_frames = {x for x in validation_frames if int(x) <= upper_bound}

    # Check if training or validation sets are empty
    if not training_frames or not validation_frames:
        raise ValueError("Arguments resulted in empty training or validation set.")

    client_ids = list(range(num_clients))
    training_dataset_list = []

    for client_id in tqdm(client_ids, desc="Creating datasets"):
        client_frame_ids = get_random_sized_subset(
            list(training_frames), client_id, num_clients, seed
        )
        training_dataset_list.append(
            create_dataset(
                zod_frames, client_frame_ids, bounding_box_format=bounding_box_format
            )
        )

    # training_dataset = create_dataset(zod_frames, training_frames)
    validation_dataset = create_dataset(
        zod_frames, validation_frames, bounding_box_format=bounding_box_format
    )

    return training_dataset_list, validation_dataset
