import torch
import torch.nn as nn
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server.strategy import FedAvg
import yaml
from typing import Tuple, Optional
from task import Net, get_weights, load_data, test, set_weights

class LogAccuracyStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _, self.testloader = load_data(
            partition_id=0,
            num_partitions=1,
            batch_size=32,
        )
        self.net = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(
        self,
        rnd: int,
        parameters,
    ) -> Optional[Tuple[float, Metrics]]:
        ndarrays = parameters_to_ndarrays(parameters)
        set_weights(self.net, ndarrays)
        loss, accuracy = test(self.net, self.testloader, self.device)

        print(f"Round {rnd} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return loss, {"accuracy": accuracy,"loss":loss}


if __name__ == "__main__":

    with open("global_server_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    server_cfg = config["server"]
    strategy_cfg = config["strategy"]

    # Read from config
    num_rounds = server_cfg["global_rounds"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Build strategy params dynamically
    strategy = LogAccuracyStrategy(
        fraction_fit=strategy_cfg["fraction_fit"],
        fraction_evaluate=strategy_cfg["fraction_evaluate"],
        min_fit_clients=strategy_cfg["min_fit_clients"],
        min_evaluate_clients=strategy_cfg["min_evaluate_clients"],
        min_available_clients=strategy_cfg["min_available_clients"],
        initial_parameters=parameters,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=server_cfg["address"],
        config=fl.server.ServerConfig(num_rounds=server_cfg["global_rounds"]),
        strategy=strategy,
    )
