import sys
import scipy.ndimage
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

if os.path.isfile('textures.npy'):
    print('==> Load previously saved textures data')
    textures = np.load('textures.npy')
else:
    print('==> Loading data')
    num_textures = 689 # hard-coded, since 689 textures
    textures = np.zeros((512,512,num_textures))
    for i in range(num_textures):
        fn = '/scratch/mad573/textures/' + str(i) + '.jpg'
        textures[:,:,i] = scipy.ndimage.imread(fn, flatten=True)/255
    
    np.save('textures.npy',textures)


random = np.random.rand(512,512,np.shape(textures)[2])

print('==> mean centering data')
pop_mean = np.mean(np.concatenate(random,textures,axis=2))
random = random - pop_mean
textures = textures - pop_mean

pop_std = np.std(np.concatenate(random,textures,axis=2))
random = random/pop_std
textures = textures/pop_std


#plt.imshow(unlabeled[:,:,0], cmap=plt.get_cmap('gray'))
#plt.show()

if len(sys.argv)>1:
    filter_size = int(sys.argv[1])
    step_size = int(sys.argv[2])
    out_dimension = int(sys.argv[3])
    LR = float(sys.argv[4])
    n_samples = int(sys.argv[5])
else:
    filter_size = 512
    step_size = 1
    out_dimension = 20
    LR = .00000001
    n_samples = 500

nonlinearity = hl.TANH
print('==> Training')
random_k = fl.Train(random[:,:,:n_samples], filter_size, step_size, out_dimension, LR, nonlinearity)
textures_k = fl.Train(textures[:,:,:n_samples], filter_size, step_size, out_dimension, LR, nonlinearity)

np.save('textures-k.npy',textures_k)

output = fl.ImageReconstruction(unlabeled[:,:,0], weights, filter_size, step_size, nonlinearity)
plt.imshow(output, cmap=plt.get_cmap('gray'))
plt.show()



print('==> Classification performance')
sigma = 0.5*(random_k + textures_k)

sig_inv = np.linalg.pinv(sigma)

diff_mean = np.mean(random[:,:,:n_samples], axis=0) - np.mean(textures[:,:,:n_samples], axis=0)

test = np.concatenate(random[:,:,501:600], random[:,:,501:600], axis=2)
y = np.zeros((200,1))
y[:100]=1
shuff = np.permutation(200)
test = test[:,:,shuff]
y = y[shuff]
corr = 0

for i in range(200):
    yhat = np.sign(np.dot(np.dot(sig_inv,diff_mean).T, test[:,:,i]))
    if (yhat == y[i]):
        corr = corr+1

pc = corr/200

print('==> Percent Correct')
print(pc)





