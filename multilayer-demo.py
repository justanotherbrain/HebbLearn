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
    fs = int(sys.argv[1])
    ss = int(sys.argv[2])
    od = int(sys.argv[3])
    lr = float(sys.argv[4])
    n_samples = int(sys.argv[5])
else:
    fs = [6, 4]
    ss = [3, 2]
    od = [10, 8]
    lr = [1, 1]
    n_samples = 100000

nl = [hl.TANH, hl.TANH]

ml = hl.MultilayerGHA(num_layers=2, filter_size=fs, step_size=ss, out_dim=od, LR=lr, nonlinearity=nl)

print('==> Training')
layers = ml.Train(unlabeled[:,:,:n_samples])

np.save('multi-layers.npy',layers)

output = ml.ImageReconstruction(unlabeled[:,:,0], layers)
#plt.imshow(output[:,:,0], cmap=plt.get_cmap('gray'))
#plt.show()





