import numpy as np

# save some dummy weights
np.savez('model.npz',
    conv_kernel  = np.random.randn(3, 3),
    linear1_w    = np.random.randn(676, 128),
    linear1_b    = np.random.randn(128),
)

# inspect it
data = np.load('model.npz')
print(data.files)          # ['conv_kernel', 'linear1_w', 'linear1_b']
print(data['conv_kernel']) # prints the 3x3 array
print(data['conv_kernel'].shape)  # (3, 3)