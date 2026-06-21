import torch

# # Introduction to Tensors
#
# # Scalar
# scalar = torch.tensor(7)
# print(scalar.ndim)
#
# # Vector
# vector = torch.tensor([7, 7])
# print(vector)
# print(vector.ndim)
# print(vector.shape)
#
# # MATRIX
# MATRIX = torch.tensor([[7, 8],
#                       [9, 10]])
# print(MATRIX.ndim)
# print(MATRIX.shape)
#
# # TENSOR
# TENSOR = torch.tensor([[[1, 2, 3],
#                         [4, 5, 6]]])
# print(TENSOR)
# print(TENSOR.ndim)
# print(TENSOR.shape)

data = [[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]

TENSOR = torch.tensor(data)
print(f"Shape of tensor: {TENSOR.shape}")
print(f"Datatype of tensor: {TENSOR.dtype}")
print(f"Device tensor is stored on: {TENSOR.device}")

# Reflection
# Tensors hold values in list-like structures
# Scalar is a type of tensor that is a single value
# Vector is a type of Tensor with direction and magnitude
# Tensor