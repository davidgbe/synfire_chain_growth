import aux
import numpy as np

def gen():
	for i in range(10):
		a = i * np.ones((2, 5, 2))
		yield a

ba = aux.BatchedArray(gen(), 5, 10, 1)

for i in range(10):
	print(ba[:, 5*i: 5*(i+1), :])

ba = aux.BatchedArray(gen(), 5, 10, 1)

