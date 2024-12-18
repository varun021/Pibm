import numpy as np

my_array = np.random.uniform(0, 1, size=(5, 5))
print("Original Array:")
print(my_array)

normalized_array = (my_array - np.min(my_array)) / (np.max(my_array) - np.min(my_array))

rounded_array = np.round(normalized_array, 2)
print("\nNormalized and Rounded Array:")
print(rounded_array)