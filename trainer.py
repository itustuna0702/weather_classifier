import torch
import os
from tqdm import tqdm
from model.modeling_output import logits_to_labels
from utils import evaluate_model

def train(model, train_loader, val_loader, loss_fn, optimizer, device, config, class_names):
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = logits_to_labels(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        evaluate_model(all_labels, all_preds, class_names, epoch+1, config['checkpoint_dir'])
        torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], f"model_epoch_{epoch+1}.pth"))
