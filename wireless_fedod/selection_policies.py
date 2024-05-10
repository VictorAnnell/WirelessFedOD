import numpy as np


def random_agent_selection(self, cars):
    """Select half of the clients randomly."""
    clients = np.random.choice(cars, len(cars) // 2, replace=False)
    if len(clients) == 0:
        clients = cars
    return clients


def loss_based_selection(self, cars):
    """Select half of the clients based on the loss."""
    # Filter out clients without loss values
    clients = [car for car in cars if car.loss is not None]
    clients = sorted(clients, key=lambda x: x.loss, reverse=True)
    # Add back clients without loss values
    clients += [car for car in cars if car.loss is None]
    clients = clients[: len(clients) // 2]
    # If no clients have loss, use random selection
    if len(clients) == 0:
        clients = random_agent_selection(self, cars)
    return clients


def deviation_based_selection(self, cars):
    """Select half of the clients based on the deviation."""
    clients = sorted(cars, key=lambda x: x.deviation, reverse=True)
    clients = clients[: len(clients) // 2]
    return clients


def bit_rate_based_selection(self, cars):
    """Select half of the clients based on the bit rate."""
    clients = sorted(cars, key=lambda x: x.bit_rate, reverse=True)
    clients = clients[: len(clients) // 2]
    return clients
