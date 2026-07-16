from torch.utils.data import DataLoader

import torch
import torchmetrics

def train(model: torch.nn.Module,
          device: torch.device,
          loss_fn: torch.nn.CrossEntropyLoss,
          optimizer: torch.optim.AdamW,
          acc_fn: torchmetrics.classification.MulticlassAccuracy,
          train_dataloader: DataLoader):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        y_logits = model(X)
        y_preds = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y)
        acc = acc_fn(y_preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc.item()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return model, train_loss, train_acc

def test(model: torch.nn.Module,
         device: torch.device,
         loss_fn: torch.nn.CrossEntropyLoss,
         acc_fn: torchmetrics.classification.MulticlassAccuracy,
         test_dataloader: DataLoader):
    with torch.inference_mode():
        model.eval()
        train_loss, train_acc = 0.0, 0.0
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=1)
            loss = loss_fn(y_logits, y)
            acc = acc_fn(y_preds, y)

            train_loss += loss.item()
            train_acc += acc.item()
        train_loss /= len(test_dataloader)
        train_acc /= len(test_dataloader)

    return train_loss, train_acc