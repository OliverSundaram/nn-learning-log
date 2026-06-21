import torch

# float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
#                                dtype=None, # what datatype is the tensor (e.g. float32)
#                                device=None) # what device is the tensor on (e.g. cpu, gpu)
#
# print(float_32_tensor)
# print(float_32_tensor.dtype)
#
# float_16_tensor = float_32_tensor.type(torch.float16) # Cast float_32_tensor into a float16 tensor
# print(float_16_tensor.dtype)

unsigned_int_8_tensor = torch.tensor(data=[-1, 1, 2])

print(unsigned_int_8_tensor[1])

# Reflection
# Datatypes for tensors are important for saving run time, or getting more complex numbers
# The higher bit the more space it takes