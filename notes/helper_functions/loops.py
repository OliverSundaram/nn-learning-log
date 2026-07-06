import torch
import torchvision

def train(model: torch.nn.Module,
          loss_fn: torch.nn.CrossEntropyLoss | torch.nn.BCEWithLogitsLoss,
          optimizer: torch.optim.Optimizer,
          accuracy_fn,
          train_dataloader: torch.utils.data.DataLoader):
    """
    Performs a training step on *model*.
    """
    model.train()
    train_loss, train_acc = 0, 0
    for batch in train_dataloader:
        X, y = batch
        y_logits = model(X)
        y_preds = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y, y_preds)
        train_loss += loss.item()
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss, train_acc

def test(model: torch.nn.Module,
          loss_fn: torch.nn.CrossEntropyLoss | torch.nn.BCEWithLogitsLoss,
          accuracy_fn,
          test_dataloader: torch.utils.data.DataLoader):
    """
    Performs a testing loop to evaluate **model**
    """
    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for batch in test_dataloader:
            X, y = batch
            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=1)
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y, y_preds)
            test_loss += loss.item()
            test_acc += acc
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    return test_loss, test_acc