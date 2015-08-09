import sys
import os.path
import HebbLearn as hl
import numpy as np
import matplotlib.pyplot as plt
try:
    import h5py
except:
    print('h5py cannot be loaded - may cause error')
    pass


fl = hl.NonlinearGHA()

if os.path.isfile('processed_data.npy'):
    print('==> Load previously saved (preprocessed) data')
    unlabeled = np.load('processed_data.npy')
else:
    print('==> Loading data')
    f = h5py.File('/scratch/mad573/stl10/unlabeled.mat')

    u = f['X'][()]

    temp = np.reshape(u, (3,96,96,100000))
    temp = np.swapaxes(temp,0,2)


    unlabeled = np.zeros((96,96,100000))

    print('==> Preprocessing data')
    for i in range(100000):
        unlabeled[:,:,i] = hl.rgb2gray(temp[:,:,:,i])
        if np.max(unlabeled[:,:,i])>1:
            unlabeled[:,:,i] = unlabeled[:,:,i]/255
    
    np.save('processed_data.npy',unlabeled)


print('==> mean centering data')
pop_mean = np.mean(unlabeled)
unlabeled = unlabeled - pop_mean

pop_std = np.std(unlabeled)
unlabeled = unlabeled/pop_std


#plt.imshow(unlabeled[:,:,0], cmap=plt.get_cmap('gray'))
#plt.show()

if len(sys.argv)>1:
    filter_size = int(sys.argv[1])
    step_size = int(sys.argv[2])
    out_dimension = int(sys.argv[3])
    LR = float(sys.argv[4])
    n_samples = int(sys.argv[5])
else:
    filter_size = 8
    step_size = 1
    out_dimension = 8
    LR = 1
    n_samples = 100000

nonlinearity = hl.TANH
print('==> Training')
weights = fl.Train(unlabeled[:,:,:n_samples], filter_size, step_size, out_dimension, LR, nonlinearity)

np.save('nl-stl-dev-weights.npy',weights)

output = fl.ImageReconstruction(unlabeled[:,:,0], weights, filter_size, step_size, nonlinearity)
plt.imshow(output, cmap=plt.get_cmap('gray'))
plt.show()





