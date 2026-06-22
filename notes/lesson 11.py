import torch

# Finding the positional min and max
tensor = torch.rand(2)
print(tensor)

# Find the position in the tensor that has the minimum value
min_value = tensor.argmin()
print(min_value)

# Find the position in the tensor that has the maximum value
max_value = tensor.argmax()
print(max_value)