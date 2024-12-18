import numpy as np
my_array = np.random.randint(0, 9, size=(6, 6))
print("Original array:")
print(my_array)
my_array[my_array % 2 == 0] = -1
print("Modified array:")
print(my_array)