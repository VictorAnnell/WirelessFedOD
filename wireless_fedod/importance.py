import numpy as np


def random_based_importance(self, car) -> float:
    """Return a random importance value."""
    return np.random.rand()


def loss_based_importance(self, car) -> float:
    """Return the loss of the client."""
    return car.loss


def deviation_based_importance(self, car) -> float:
    """Return the deviation of the client."""
    return car.deviation


def bit_rate_based_importance(self, car) -> float:
    """Return the bit rate of the client."""
    return car.bit_rate


def loss_combined_importance(self, car) -> float:
    from config import LEARNING_IMPORTANCE_WEIGHT, NETWORK_IMPORTANCE_WEIGHT

    """Return the combined importance of loss and bit rate."""
    return (car.loss * LEARNING_IMPORTANCE_WEIGHT) + (car.bit_rate * NETWORK_IMPORTANCE_WEIGHT)


def deviation_combined_importance(self, car) -> float:
    from config import LEARNING_IMPORTANCE_WEIGHT, NETWORK_IMPORTANCE_WEIGHT

    """Return the combined importance of deviation and bit rate."""
    return (car.deviation * LEARNING_IMPORTANCE_WEIGHT) + (car.bit_rate * NETWORK_IMPORTANCE_WEIGHT)
