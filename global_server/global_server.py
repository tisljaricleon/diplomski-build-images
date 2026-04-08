import torch
import torch.nn as nn
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server.strategy import FedAvg
import yaml
from typing import Tuple, Optional
from task import Net, get_weights, load_data, test, set_weights, load_model, save_model


class LogAccuracyStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _, self.testloader = load_data(
            partition_id=0,
            num_partitions=1,
            batch_size=32,
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = load_model("/home/model/model.pt", self.device)

    def evaluate(
        self,
        rnd: int,
        parameters,
    ) -> Optional[Tuple[float, Metrics]]:
        ndarrays = parameters_to_ndarrays(parameters)
        set_weights(self.net, ndarrays)
        loss, accuracy = test(self.net, self.testloader, self.device)
        save_model(self.net, "/home/model/model.pt")
        print(f"Round {rnd} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, {"accuracy": accuracy, "loss": loss}


if __name__ == "__main__":

    with open("global_server_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    server_cfg = config["server"]
    strategy_cfg = config["strategy"]

    num_rounds = server_cfg["global_rounds"]

    pretrained_model = load_model("/home/model/model.pt", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    ndarrays = get_weights(pretrained_model)
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = LogAccuracyStrategy(
        fraction_fit=strategy_cfg["fraction_fit"],
        fraction_evaluate=strategy_cfg["fraction_evaluate"],
        min_fit_clients=strategy_cfg["min_fit_clients"],
        min_evaluate_clients=strategy_cfg["min_evaluate_clients"],
        min_available_clients=strategy_cfg["min_available_clients"],
        initial_parameters=parameters,
    )

    fl.server.start_server(
        server_address=server_cfg["address"],
        config=fl.server.ServerConfig(num_rounds=server_cfg["global_rounds"]),
        strategy=strategy,
    )
