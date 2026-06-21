import torch

# Matrix Multiplication EXPANDED

# Common errors in deep learning: Tensor shape errors
# Main rules for matrix multiplication:
# 1. the inner dimensions must match:
# (3, 2) @ (3, 2) wont work
# (2, 3) @ (3, 2) will work

print(torch.matmul(torch.rand(3, 2), torch.rand(2, 3)))

# 2. The resulting matrix has the shape of the outer dimensions

print(torch.matmul(torch.rand(3, 2), torch.rand(2, 3)).shape) # Prints 3, 3



