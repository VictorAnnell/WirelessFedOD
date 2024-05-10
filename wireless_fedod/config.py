import math
import os

from dotenv import load_dotenv
from models import *  # noqa: F403
from importance import *  # noqa: F403
from zod.anno.object import OBJECT_CLASSES

load_dotenv()

# Dataset configuration
DATASET_ROOT = os.getenv("DATASET_ROOT", "../datasets")
DATASET_VERSION = os.getenv("DATASET_VERSION", "full")
DATASET_MAX_IMAGES = os.getenv("DATASET_MAX_IMAGES", 200) if DATASET_VERSION == "full" else 'none'
try:
    DATASET_MAX_IMAGES = int(DATASET_MAX_IMAGES)
except ValueError:
    if DATASET_MAX_IMAGES.lower() == "none":
        DATASET_MAX_IMAGES = None
    else:
        raise ValueError(f"Invalid value for DATASET_MAX_IMAGES: {DATASET_MAX_IMAGES}")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
SHUFFLE_BUFFER_SIZE = int(os.getenv("SHUFFLE_BUFFER_SIZE", BATCH_SIZE * 10))
OBJECT_CLASSES = OBJECT_CLASSES
CLASS_MAPPING = dict(zip(range(len(OBJECT_CLASSES)), OBJECT_CLASSES))

# Model configuration
# Set model creation function:
MODEL_FN = os.getenv("MODEL_FN", yolov8_model_fn.__name__)  # noqa: F405
try:
    MODEL_FN = globals()[MODEL_FN]
except KeyError:
    raise ValueError(f"Model {MODEL_FN} not found in models.py")

# Simulator configuration
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", 5))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", 1))
STEPS_PER_LOCAL_EPOCH = os.getenv("STEPS_PER_LOCAL_EPOCH", 'none')
try:
    STEPS_PER_LOCAL_EPOCH = int(STEPS_PER_LOCAL_EPOCH)
except ValueError:
    if STEPS_PER_LOCAL_EPOCH.lower() == "none":
        STEPS_PER_LOCAL_EPOCH = None
    else:
        raise ValueError(f"Invalid value for STEPS_PER_LOCAL_EPOCH: {STEPS_PER_LOCAL_EPOCH}")
SIMULATION_ID = os.getenv("SIMULATION_ID", None)
# Set importance function:
IMPORTANCE_FN = os.getenv("IMPORTANCE_FN", "random_based_importance")
try:
    IMPORTANCE_FN = globals()[IMPORTANCE_FN]
except KeyError:
    raise ValueError(f"Importance function {IMPORTANCE_FN} not found in importance.py")
LEARNING_IMPORTANCE_WEIGHT = float(os.getenv("LEARNING_IMPORTANCE_WEIGHT", 0.5))
NETWORK_IMPORTANCE_WEIGHT = float(os.getenv("NETWORK_IMPORTANCE_WEIGHT", 0.5))
# Scale weights to sum to 1
importance_weight_sum = LEARNING_IMPORTANCE_WEIGHT + NETWORK_IMPORTANCE_WEIGHT
LEARNING_IMPORTANCE_WEIGHT /= importance_weight_sum
NETWORK_IMPORTANCE_WEIGHT /= importance_weight_sum

# Wireless network configuration
BANDWIDTH = 20e6  # in Hz
SIGNAL_POWER = 1 # in Watt
NOISE_POWER = 1 # in Watt
SNR = SIGNAL_POWER/NOISE_POWER

ALTITUTE_VEHICLE = 1.6  # in m
ALTITUTE_BS = 25  # in m
SPEED_OF_LIGHT = 3 * math.pow(10, 8)  # in m/s
FREQUENCY = 3.5 * math.pow(10, 9)  # in Hz
PATH_LOSS_EXPONENT = 3.7
HPBW_BS = 25
SLL_BS = 19.1
BEAMS_AZIMUTH = [15, 45, -15, -45]
BEAMS_ELEVATION = [16.8]  # , 46.8, 76.8]
