import torch

def logits_to_labels(logits):
    return torch.argmax(logits, dim=1)
