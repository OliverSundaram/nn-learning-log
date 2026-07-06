import torch

# Finding tensor min, max, mean, sum, etc. (tensor aggregation)

# New tensor
tensor = torch.arange(0, 101, 10)

# Find the min
min = tensor.min()
min = torch.min(tensor)

# Find the max
max = tensor.max()
max = torch.max(tensor)

# # Find the mean
# mean = tensor.mean()
# mean = torch.mean(tensor)

# Got error after running .mean()
# The mean() takes in a floating point or complex dtype

mean = torch.mean(tensor.type(torch.float32))

# Find the sum
sum = tensor.sum()
sum = torch.sum(tensor)

print(min)
print(max)
print(mean)
print(sum)