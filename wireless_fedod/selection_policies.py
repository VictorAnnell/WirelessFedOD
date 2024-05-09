import numpy as np


def random_agent_selection(self, cars):
    """Select half of the clients randomly."""
    clients = np.random.choice(cars, len(cars) // 2, replace=False)
    if len(clients) == 0:
        clients = cars
    return clients


def loss_based_selection(self, cars):
    """Select half of the clients based on the loss."""
    clients = sorted(cars, key=lambda x: x.loss, reverse=True)
    clients = clients[: len(clients) // 2]
    return clients


def deviation_based_selection(self, cars):
    """Select half of the clients based on the deviation."""
    clients = sorted(cars, key=lambda x: x.deviation, reverse=True)
    clients = clients[: len(clients) // 2]
    return clients
