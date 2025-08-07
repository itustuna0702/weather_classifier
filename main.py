import yaml
import torch
import os
from model.models import get_model
from model.losses import get_loss
from data_utils import get_dataloaders
from trainer import train
from torchvision import datasets

if __name__ == '__main__':
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders("weather_dataset", config['batch_size'], config['img_size'], config['train_ratio'], config['val_ratio'])
    class_names = datasets.ImageFolder("weather_dataset").classes

    model = get_model(config['model_name'], config['num_classes']).to(device)
    loss_fn = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train(model, train_loader, val_loader, loss_fn, optimizer, device, config, class_names)
