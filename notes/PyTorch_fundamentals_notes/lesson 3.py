import torch

# # creating tensors with zeros and 1's
# # Tensor with zeros
# zero = torch.zeros(size=(3, 4))
#
# print(zero)
#
# # Tensor with 1's
# ones = torch.ones(size=(2, 3))
# print(ones)
# print(ones.dtype)

MATRIX = torch.tensor([[1, 2, 3], [4, 5, 6]])

zeros = torch.zeros(size=(2, 1))
print(zeros)

print(zeros*MATRIX)