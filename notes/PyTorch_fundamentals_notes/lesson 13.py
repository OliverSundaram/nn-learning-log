import torch
# Indexing (Selecting CIFAR10_data from tensors)
# Create a tensor
x = torch.arange(1, 10).reshape(1, 3, 3)

print(x[0])
print()

print(x[0][0])
print(x[0, 0])
print()

print(x[0, 0, 2])
print(x[0][0][2])
print()

# You can use ":" to select all of a target dimension
# Get all values in the 0 dim, all in first, and all values at 0 index in 2nd dim
print(x[:, :, 0])
print()

# Get index 0 of 0 dim and 1 dim and all values of 2 dim
print(x[0, 0, :])

# print [3, 6, 9]
print(x[0, :, 2])