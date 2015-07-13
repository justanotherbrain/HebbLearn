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
	filter_size = int(sys.argv[1])
	iterations = int(sys.argv[2])
	out_dimension = int(sys.argv[3])
	LR = float(sys.argv[4])
else:
	filter_size = 8
	iterations = 2
	out_dimension = 8
	LR = .000001

weights = hl.TrainSlidingLinearGHA(goat, filter_size, out_dimension, LR, iterations)

output = hl.SlidingImageReconstruction(goat, weights, filter_size)
plt.imshow(output, cmap=plt.get_cmap('gray'))
plt.show()

hl.VisualizeFilter(weights)

print(weights)




