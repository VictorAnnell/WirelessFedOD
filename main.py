import random

import keras
from dotenv import load_dotenv

from wireless_fedod.config import MIXED_PRECISION, NUM_ROUNDS, SEED
from wireless_fedod.dataset import load_zod
from wireless_fedod.simulator import WirelessFedODSimulator

load_dotenv()


# Mixed precision
if MIXED_PRECISION:
    print("Mixed precision enabled")
    keras.mixed_precision.set_global_policy("mixed_float16")

# Sets the seed for NumPy, TensorFlow, Keras, and random
if SEED == -1:
    # Set random seed
    SEED = random.randint(0, 1000000)
print(f"Random seed: {SEED}")
keras.utils.set_random_seed(SEED)

if __name__ == "__main__":
    simulator = WirelessFedODSimulator()

    # Load ZOD dataset
    zod_train, zod_test = load_zod()

    # Set dataset
    simulator.train_data = zod_train
    simulator.test_data = zod_test
    for _ in range(NUM_ROUNDS):
        simulator.run_round()
