import torch

# Manipulating tensors (tensor operations)

# Addition
# Subtraction
# Multiplication (element-wise)
# Division
# Matrix multiplication

# Addition
tensor = torch.tensor([1, 2, 3])

print(tensor + 10)
print(tensor * 10)
print(tensor - 10)
# PyTorch inbuilt functions
print(torch.mul(tensor, 10))
print(torch.add(tensor, 5))
print(torch.div(tensor, 2))


# Matrix Multiplication (dot product)
# torch.matmul (matrix multiply)
print(torch.matmul(tensor, tensor)) # prints 14

TENSOR1 = torch.tensor([[1, 2, 3]])
TENSOR2 = torch.tensor([[3],
                        [2],
                        [1]])

print(torch.matmul(TENSOR1, TENSOR2))