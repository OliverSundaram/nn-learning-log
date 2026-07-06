import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, device: str):
    # Move model and CIFAR10_data to CPU for visualization
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Convert evaluation grid to PyTorch tensor
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    # Model predictions on the grid
    model.eval()
    with torch.inference_mode():
        grid_logits = model(grid_tensor)
        grid_preds = torch.argmax(grid_logits, dim=1).reshape(xx.shape)

    # Plot the background decision regions
    plt.contourf(xx, yy, grid_preds.numpy(), alpha=0.3, cmap=plt.cm.RdYlBu)
    # Plot the actual CIFAR10_data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")