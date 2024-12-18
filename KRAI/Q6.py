import numpy as np
my_array=np.random.randint(1,10,size=(5,5))
print(my_array)
print("Sum of each row:")
my_1d_array=np.array(np.sum(my_array,axis=0))
print(my_1d_array)