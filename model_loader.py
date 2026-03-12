import torch
from model import enhance_net_nopool

def load_model():
    model = enhance_net_nopool()
    model.load_state_dict(torch.load("snapshots/Epoch99.pth", map_location="cpu"))
    model.eval()
    return model