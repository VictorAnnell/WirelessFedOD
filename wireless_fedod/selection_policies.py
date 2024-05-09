import numpy as np


def random_agent_selection(self, cars):
    """Select half of the clients randomly."""
    clients = np.random.choice(cars, len(cars) // 2, replace=False)
    if len(clients) == 0:
        clients = cars
    return clients
