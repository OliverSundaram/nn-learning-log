import torch

# 1. Documentation reading - A big part of deep learning (and learning to code in general)
# is getting familiar with the documentation of a certain framework you're using. We'll
# be using the PyTorch documentation a lot throughout the rest of this course. So I'd
# recommend spending 10-minutes reading the following (it's okay if you don't get some
# things for now, the focus is not yet full understanding, it's awareness).
# See the documentation on torch.Tensor and for torch.cuda.

# 2 things learned from torch.Tensor
# A tensor holds only a single datatype, cannot mix
# If a tensor hold a single value, you can use .item()

# 2 things learned from torch.cuda
#


# 2. Create a random tensor with shape (7, 7).
random_tensor = torch.rand(7, 7)



# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
tensor = torch.rand(1, 7)
tensor = tensor.T
print(torch.matmul(random_tensor, tensor))

# 4. Set the random seed to 0 and do exercises 2 & 3 over again.
RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)
tensor_A = torch.rand(7, 7)
print(tensor_A)

torch.manual_seed(RANDOM_SEED)
tensor_B = torch.rand(1, 7)
print(tensor_B)
tensor_B = torch.transpose(tensor_B, 0, 1)

print(torch.matmul(tensor_A, tensor_B))

# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but
# is there a GPU equivalent? (hint: you'll need to look into the documentation
# for torch.cuda for this one). If there is, set the GPU random seed to 1234.

torch.cuda.manual_seed(RANDOM_SEED)

# 6. Create two random tensors of shape (2, 3) and send them both to the
# GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234)
# when creating the tensors (this doesn't have to be the GPU random seed).

device = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 1234

torch.manual_seed(RANDOM_SEED)
tensor_A = torch.rand(2, 3)
torch.manual_seed(RANDOM_SEED)
tensor_B = torch.rand(2, 3)

tensor_A_on_gpu = tensor_A.to(device)
tensor_B_on_gpu = tensor_B.to(device)

print(tensor_A_on_gpu)
print(tensor_B_on_gpu)

# 7. Perform a matrix multiplication on the tensors you created in 6
# (again, you may have to adjust the shapes of one of the tensors).

tensor_B_on_gpu = tensor_B_on_gpu.T
result = torch.matmul(tensor_A_on_gpu, tensor_B_on_gpu)
print(result)

# 8. Find the maximum and minimum values of the output of 7.

max = result.max()
min = result.min()

print(max, min)

# 9. Find the maximum and minimum index values of the output of 7.

max_pos = result.argmax()
min_pos = result.argmin()
print(max_pos, min_pos)

# 10. Make a random tensor with shape (1, 1, 1, 10) and then create
# a new tensor with all the 1 dimensions removed to be left with a
# tensor of shape (10). Set the seed to 7 when you create it and
# print out the first tensor and it's shape as well as the second
# tensor and it's shape.

RANDOM_SEED = 13
torch.manual_seed(RANDOM_SEED)
tensor = torch.rand(1, 1, 1, 10)
print(tensor, tensor.shape)

tensor = tensor.squeeze()
print(tensor, tensor.shape)