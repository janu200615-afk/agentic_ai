# fl_client.py
# Flower client that wraps the existing training logic
import flwr as fl
import torch
import numpy as np
from model import SmallNet, get_state_dict_bytes, load_state_dict_bytes
from utils import gen_synthetic_data, prepare_data
import argparse
import torch.nn as nn
import torch.optim as optim

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        # set model parameters, train locally, return updated parameters
        self.set_parameters(parameters)
        self._train(epochs=int(config.get("epochs", 1)))
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self._test()
        return float(loss), len(self.X_test), {"accuracy": float(acc)}

    def _train(self, epochs=1):
        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        X = torch.from_numpy(self.X_train)
        y = torch.from_numpy(self.y_train)
        for _ in range(epochs):
            opt.zero_grad()
            out = self.model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

    def _test(self):
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            out = self.model(torch.from_numpy(self.X_test))
            loss = loss_fn(out, torch.from_numpy(self.y_test))
            preds = out.argmax(dim=1).numpy()
            acc = (preds == self.y_test).mean()
        return float(loss), float(acc)

def main(agent_id=0, samples=150):
    df = gen_synthetic_data(num_samples=samples, seed=agent_id)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = SmallNet()
    client = FLClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address='127.0.0.1:8080', client=client)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--samples', type=int, default=150)
    args = parser.parse_args()
    main(agent_id=args.id, samples=args.samples)
