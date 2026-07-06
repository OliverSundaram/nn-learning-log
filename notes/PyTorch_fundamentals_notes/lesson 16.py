import torch
import numpy

# Running tensors & PyTorch objects on gpu's (making faster computations)
# GPUs = faster computations on numbers, because of CUDA + NVIDIA hardware + PyTorch



# Check for GPU access with PyTorch
# print(torch.cuda.is_available())

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Count number of gpu's
print(torch.cuda.device_count())



# Putting tensors and models_state_dict on to GPU
# create a tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])
# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

# np_array = tensor_on_gpu.numpy()
# Trying to convert tensor on gpu to numpy array causes error
# You first have to move it back to the CPU with tensor.cpu()
tensor_on_cpu = tensor_on_gpu.cpu()
np_array = tensor_on_cpu.numpy()
print(tensor_on_cpu, np_array)