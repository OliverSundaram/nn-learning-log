import torch

# TENSOR_A = torch.tensor([[1, 2],
#                          [3, 4],
#                          [5, 6]])
# # E.x. Tensor is 3x2, e.g. Days, cost of item sold
# TENSOR_B = torch.tensor([[7, 10],
#                          [8, 11],
#                          [9, 12]])
#
# # Tensor manipulation
# # Using transpose, switches axis of a tensor
# print(TENSOR_B.T)
# # E.x. Swapping tensor from days, cost to cost, dasy
# TENSOR_B = TENSOR_B.T
# print(torch.matmul(TENSOR_A, TENSOR_B))

#     E.g. Width, Height, Channels
image = torch.rand(2, 2, 3)
# Swap Width & Channels
image = image.transpose(0, 2)
# Swap Height & Width
image = image.transpose(1, 2)

print(image)

tensor = torch.rand(2, 3)
print(tensor)
print(tensor.T)