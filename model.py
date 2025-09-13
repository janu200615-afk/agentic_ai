import torch
import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self, input_dim=6, hidden=32, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def get_state_dict_bytes(model):
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

def load_state_dict_bytes(model, bts, map_location=None):
    import io
    buffer = io.BytesIO(bts)
    state = torch.load(buffer, map_location=map_location)
    model.load_state_dict(state)
    return model
