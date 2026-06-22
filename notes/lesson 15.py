import torch

# Reproducibility (trying to take the random out of random)

# To reduce the randomness in nn and PyTorch, can use a random seed

tensor_a = torch.rand(3, 4)
tensor_b = torch.rand(3, 4)

print(tensor_a)
print(tensor_b)
print(tensor_a == tensor_b)

# Make a random but reproducible tensor
# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

tensor_c = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
tensor_d = torch.rand(3, 4)

print(tensor_c)
print(tensor_d)
print(tensor_c == tensor_d)