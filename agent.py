# agent.py
import argparse
import requests
import base64
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import SmallNet, get_state_dict_bytes, load_state_dict_bytes
from utils import gen_synthetic_data, prepare_data
from sklearn.metrics import accuracy_score

SERVER = 'http://127.0.0.1:5000'

def train_local(model, X_train, y_train, epochs=3, lr=0.01):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.from_numpy(X_train)
    y = torch.from_numpy(y_train)
    for e in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X_test))
        preds = out.argmax(dim=1).numpy()
    return accuracy_score(y_test, preds)

def main(agent_id=0, samples=150, epochs=3, simulate=False):
    print(f'[Agent {agent_id}] Generating local data...')
    df = gen_synthetic_data(num_samples=samples, seed=agent_id)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # get current global model
    try:
        r = requests.get(SERVER + '/get_global', timeout=5)
        if r.status_code == 200:
            model_b64 = r.content.decode('utf-8')
            global_bytes = base64.b64decode(model_b64.encode('utf-8'))
            model = SmallNet()
            load_state_dict_bytes(model, global_bytes, map_location=torch.device('cpu'))
            print(f'[Agent {agent_id}] Loaded global model from server')
        else:
            model = SmallNet()
    except Exception as e:
        print('[Agent] Could not reach server, starting from scratch', e)
        model = SmallNet()

    before = evaluate(model, X_test, y_test)
    print(f'[Agent {agent_id}] Accuracy before local train: {before:.3f}')

    model = train_local(model, X_train, y_train, epochs=epochs)

    after = evaluate(model, X_test, y_test)
    print(f'[Agent {agent_id}] Accuracy after local train: {after:.3f}')

    # send update to server
    bts = get_state_dict_bytes(model)
    b64 = base64.b64encode(bts).decode('utf-8')
    payload = {
        'agent_id': agent_id,
        'meta': {'samples': len(X_train)},
        'model': b64
    }
    try:
        resp = requests.post(SERVER + '/post_update', json=payload, timeout=5)
        print(f'[Agent {agent_id}] Server response: {resp.json()}')
    except Exception as e:
        print('[Agent] Failed to send update', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--samples', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--simulate', action='store_true')
    args = parser.parse_args()
    main(agent_id=args.id, samples=args.samples, epochs=args.epochs, simulate=args.simulate)
