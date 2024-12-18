import numpy as np

deep_array = np.arange(1, 13).reshape(3, 4)

print("Original array:")
print(deep_array)

new_array = deep_array.reshape(4, 3)

print("\nReshaped array:")
print(new_array)