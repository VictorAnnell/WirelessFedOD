from dataset import load_zod
from models import yolov8_model_fn
from simulator import WirelessFedODSimulator

if __name__ == "__main__":
    simulator = WirelessFedODSimulator(num_clients=5)

    # Load ZOD dataset
    zod_train, zod_test = load_zod(
        # version="mini",
        version="full",
        upper_bound=150,
    )

    # Set dataset, model_fn, and agent_selection_fn
    simulator.train_data = zod_train
    simulator.test_data = zod_test
    simulator.model_fn = yolov8_model_fn
    simulator.num_clients = 1  # For testing
    simulator.local_epochs = 1  # For testing
    simulator.steps_per_local_epoch = 1  # For testing
    # while simulator.metrics['loss'] > 1:
    simulator.run_round()
