import torch

# Getting attributes from tensors
# Get datatype with .dtype
# Get shape with .shape
# Get device with .device

some_tensor = torch.rand(3, 4)
print(some_tensor)

print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device of tensor: {some_tensor.device}")

tensor = torch.rand(size=(2, 4), dtype=torch.float16, device="cpu")
print(tensor)
print(tensor.dtype)
print(tensor.shape)
print(tensor.device)