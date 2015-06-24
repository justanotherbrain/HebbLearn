import sys
import HebbLearn as hl
import numpy as np

goat = hl.rgb2gray(hl.LoadImage('goat.jpg'))
if np.max(goat)>1:
	goat = goat/255

plt.imshow(goat, cmap=plt.get_cmap('gray'))
plt.show()

if len(sys.argv)>0:
	filter_size = sys.argv[0]
	iterations = sys.argv[1]
	out_dimension = sys.argv[2]
else:
	filter_size = 8
	iterations = 2
	out_dimension = 8

weights = hl.TrainSlidingLinearGHA(goat, filter_size, out_dimension, iterations)

output = hl.ReconstructSlidingLinearGHA(goat, weights, filter_size)
plt.imshow(output, cmap=plt.get_cmap('gray'))
plt.show()

print(weights)






