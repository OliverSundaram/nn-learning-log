import torch

# Random Tensors
# random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.size())
print(random_tensor.ndim)

# random tensor similar to image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, color channels

TENSOR = torch.rand(1, 2, 2, 2)
print(TENSOR)