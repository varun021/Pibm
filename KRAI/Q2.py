import numpy as np
my_array=np.random.randint(1,11,size=(5,5))
print(my_array)
my_max=np.max(my_array)
print(my_max)
print(np.where(my_array==my_max))