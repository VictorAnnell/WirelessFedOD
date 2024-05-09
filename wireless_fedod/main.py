from dataset import load_zod
from dotenv import load_dotenv
from models import yolov8_model_fn
from selection_policies import deviation_based_selection, loss_based_selection, random_agent_selection
from simulator import WirelessFedODSimulator

load_dotenv()


def test_policies():
    # Load ZOD dataset
    zod_train, zod_test = load_zod()
    for policy in [random_agent_selection, loss_based_selection, deviation_based_selection]:
        simulator = WirelessFedODSimulator(num_clients=10, simulation_id=policy.__name__)
        simulator.train_data = zod_train
        simulator.test_data = zod_test
        simulator.model_fn = yolov8_model_fn
        simulator.agent_selection_fn = policy
        for _ in range(3):
            simulator.run_round()


if __name__ == "__main__":
    simulator = WirelessFedODSimulator(num_clients=5)

    # Load ZOD dataset
    zod_train, zod_test = load_zod()

    # Set dataset, model_fn, and agent_selection_fn
    simulator.train_data = zod_train
    simulator.test_data = zod_test
    # simulator.model_fn = yolov8_model_fn
    # simulator.num_clients = 1  # For testing
    # simulator.local_epochs = 1  # For testing
    # simulator.steps_per_local_epoch = 1  # For testing
    # while simulator.metrics['loss'] > 1:
    simulator.run_round()
