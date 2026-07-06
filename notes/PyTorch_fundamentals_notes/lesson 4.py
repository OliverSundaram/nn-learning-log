import torch

# range of tensors
one_to_ten = torch.arange(1, 11)
print(one_to_ten)

# creating tensors like
ten_zeros = torch.zeros_like(one_to_ten)
print(ten_zeros)