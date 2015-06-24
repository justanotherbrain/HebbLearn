import sys
import HebbLearn as hl
import numpy as np
import matplotlib.pyplot as plt

goat = hl.rgb2gray(hl.LoadImage('goat.jpg'))
if np.max(goat)>1:
	goat = goat/255

plt.imshow(goat, cmap=plt.get_cmap('gray'))
plt.show()

if len(sys.argv)>1:
	filter_size = sys.argv[1]
	iterations = sys.argv[2]
	out_dimension = sys.argv[3]
	LR = sys.argv[4]
else:
	filter_size = 8
	iterations = 2
	out_dimension = 8
	LR = .000001

weights = hl.TrainSlidingLinearGHA(goat, filter_size, out_dimension, LR, iterations)

output = hl.ReconstructSlidingLinearGHA(goat, weights, filter_size)
plt.imshow(output, cmap=plt.get_cmap('gray'))
plt.show()

print(weights)




