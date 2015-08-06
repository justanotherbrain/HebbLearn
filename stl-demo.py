import sys
import HebbLearn as hl
import numpy as np
import matplotlib.pyplot as plt
import h5py

fl = hl.FixedLinearGHA()

print('==> Loading data')
f = h5py.File('/scratch/mad573/stl10/unlabeled.mat')

u = f['X'][()]

temp = np.reshape(u, (3,96,96,100000), order='F')
temp = np.swapaxes(temp,0,2)


unlabeled = np.zeros((96,96,100000))

print('==> Preprocessing data')
for i in range(100000):
    unlabeled[:,:,i] = hl.rgb2gray(temp[i,:,:,:])
    if np.max(unlabeled[:,:,i])>1:
        unlabeled[:,:,i] = unlabeled[:,:,i]/255

plt.imshow(unlabeled[:,:,0], cmap=plt.get_cmap('gray'))
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

print('==> Training')
weights = fl.Train(unlabeled, filter_size, out_dimension, LR)

output = fl.ImageReconstruction(unlabeled[:,:,0], weights, filter_size)
plt.imshow(output, cmap=plt.get_cmap('gray'))
plt.show()

#fl.VisualizeFilter(weights)

#print(weights)




