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
