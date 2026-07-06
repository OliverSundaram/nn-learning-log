import torch
import numpy as np

# Pytorch tensors and NumPy

# Numpy is a popular scientific Python numerical computing library.
# Because of this, PyTorch has functionality to interact with it.
# * Data in NumPy, want in PyTorch tensor -> torch.from_numpy(ndarray)
# * Pytorch tensor -. Numpy -> torch.Tensor.numpy()

# Default numpy array datatype is float 64
array = np.arange(1.0, 8.0)
# Default PyTorch tensor datatype is float 32
tensor = torch.from_numpy(array) # Pytorch keeps the numpy dtype
print(array, tensor)

array += 1 # Only effects numpy array
print(array, tensor)

# Tensor to numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(numpy_tensor.dtype)

tensor = torch.add(tensor, 1)
print(tensor)
print(numpy_tensor)