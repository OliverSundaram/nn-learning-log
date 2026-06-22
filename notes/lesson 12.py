import torch

# Reshaping, stacking, squeezing, and unsqueezing tensors

# Reshaping - reshapes an input tensor into a defined shape
# View - Return a view of an input tensor of a certain shape but keep the same memory as the original tensor
# Stacking - Combine multiple tensors on top of eath other (vstack) or side by side (hstack)
# Squeezing - Removes all 1 dimensions from a tensor
# Unsqueeze - Adds a 1 dimension to a target tensor
# Permute - Return a view of the input with dimensions permuted (swapped) in a certain way

# Create a tensor
tensor = torch.arange(1., 3.)

print(tensor)
print(tensor.shape)
print()


# Add an extra dimension, has to be compatible. A tensor of size 9 cannot reshape into a tensor of size 1x7 -> 7 elements
tensor_reshaped = tensor.reshape(1, 9)
print(tensor_reshaped)
print(tensor_reshaped.shape)
print()


# Change the view. Returns a new tensor that is reshaped but shares the same memory, so editing 1 effects the other
z = tensor.view(1, 9)
print(z)
print(z.shape)
print()


# Stack tensors ontop of each other
tensor_stacked = torch.stack([tensor, tensor], dim=0)
# dim=0 means stack them horizontally
# dim=1 means stack them vertically
print(tensor_stacked)
print(tensor_stacked.shape)
print()

tensor_stacked = torch.vstack([tensor, tensor])
print(tensor_stacked)
print(tensor_stacked.shape)
print()

tensor_stacked = torch.hstack([tensor, tensor])
print(tensor_stacked)
print(tensor_stacked.shape)
print()

tensor_stacked = torch.dstack([tensor, tensor])
print(tensor_stacked)
print(tensor_stacked.shape)
print()

tensor = torch.rand(2, 1)
print(tensor)
print(tensor.shape)
print()

# Removes any 1's in a tensors shape
tensor_squeezed = tensor.squeeze()
print(tensor_squeezed)
print(tensor_squeezed.shape)
print()

# Adds a single dimension to a target tensor at a specific dim
tensor_unsqueezed = tensor.unsqueeze(1)
print(tensor_unsqueezed)
print(tensor_unsqueezed.shape)
print()

# Switches a tensors dimensions, similar to transpose
tensor = torch.rand(4, 4, 3)
tensor_permuted = torch.permute(tensor, (2, 1, 0))
print(tensor)
print(tensor_permuted)