import os

from dotenv import load_dotenv
from models import *  # noqa: F403
from selection_policies import *  # noqa: F403
from zod.anno.object import OBJECT_CLASSES

load_dotenv()

# Dataset configuration
DATASET_ROOT = os.getenv("DATASET_ROOT", "../datasets")
DATASET_VERSION = os.getenv("DATASET_VERSION", "mini")
DATASET_MAX_IMAGES = os.getenv("DATASET_MAX_IMAGES", 200) if DATASET_VERSION == "full" else None
if DATASET_MAX_IMAGES is not None:
    DATASET_MAX_IMAGES = int(DATASET_MAX_IMAGES)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
SHUFFLE_BUFFER_SIZE = int(os.getenv("SHUFFLE_BUFFER_SIZE", BATCH_SIZE * 10))
OBJECT_CLASSES = OBJECT_CLASSES
CLASS_MAPPING = dict(zip(range(len(OBJECT_CLASSES)), OBJECT_CLASSES))

# Model configuration
# Set model creation function:
MODEL_FN = os.getenv("SELECTION_POLICY", yolov8_model_fn.__name__)  # noqa: F405
try:
    MODEL_FN = globals()[MODEL_FN]
except KeyError:
    raise ValueError(f"Selection policy {MODEL_FN} not found in selection_policies.py")

# Simulator configuration
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", 5))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", 1))
STEPS_PER_LOCAL_EPOCH = os.getenv("STEPS_PER_LOCAL_EPOCH", None)
if STEPS_PER_LOCAL_EPOCH is not None:
    STEPS_PER_LOCAL_EPOCH = int(STEPS_PER_LOCAL_EPOCH)
SIMULATION_ID = os.getenv("SIMULATION_ID", None)
# Set agent selection policy function:
AGENT_SELECTION_FN = os.getenv("SELECTION_POLICY", "random_agent_selection")
try:
    AGENT_SELECTION_FN = globals()[AGENT_SELECTION_FN]
except KeyError:
    raise ValueError(f"Selection policy {AGENT_SELECTION_FN} not found in selection_policies.py")
