# fl_server.py
# Simple Flower server for federated learning (FedAvg)
import flwr as fl
import torch
from model import SmallNet

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)

if __name__ == "__main__":
    # Start Flower server with default strategy (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=0.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(server_address='0.0.0.0:8080', config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)
