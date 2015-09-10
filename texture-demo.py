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
num_textures = 688
if os.path.isfile('textures.npy'):
    print('==> Load previously saved textures data')
    textures = np.load('textures.npy')
else:
    print('==> Loading data')
    textures = np.zeros((512,512,num_textures))
    for i in range(num_textures):
        fn = '/home/rabadi/data/textures/' + str(i) + '.jpg'
        try:
            textures[:,:,i] = scipy.ndimage.imread(fn, flatten=True)/255
        except:
            print('dimensionality miss-match - fixing')
            tmp = scipy.ndimage.imread(fn, flatten=True)/255
            if (np.shape(tmp)[0] < 512):
                tmp = np.concatenate((tmp, np.random.rand(512-np.shape(tmp)[0],np.shape(tmp)[1])), axis=0)
            if (np.shape(tmp)[1] < 512):
                tmp = np.concatenate((tmp, np.random.rand(512, 512-np.shape(tmp)[1])), axis=1)
            textures[:,:,i] = tmp
    
    np.save('textures.npy',textures)


random = np.random.rand(512,512,np.shape(textures)[2])
random = random/np.max(random) # make sure all normalized

print('==> mean centering data')
pop_mean = np.mean(np.concatenate((random,textures),axis=2))
random = random - pop_mean
textures = textures - pop_mean

pop_std = np.std(np.concatenate((random,textures),axis=2))
random = random/pop_std
textures = textures/pop_std


#plt.imshow(textures[:,:,0], cmap=plt.get_cmap('gray'))
#plt.show()

if len(sys.argv)>1:
    filter_size = int(sys.argv[1])
    step_size = int(sys.argv[2])
    out_dimension = int(sys.argv[3])
    LR = float(sys.argv[4])
    n_samples = int(sys.argv[5])
else:
    filter_size = 512
    step_size = 512
    out_dimension = 1
    LR = 1
    n_samples = 500

nonlinearity = hl.TANH
#print('==> Training')
#random_k = fl.Train(random[:,:,:n_samples], filter_size, step_size, out_dimension, LR, nonlinearity)
#textures_k = fl.Train(textures[:,:,:n_samples], filter_size, step_size, out_dimension, LR, nonlinearity)

#np.save('textures-k.npy',textures_k)

#output = fl.ImageReconstruction(textures[:,:,0], textures_k, filter_size, step_size, nonlinearity)
#plt.imshow(output, cmap=plt.get_cmap('gray'))
#plt.show()

print('==> Classification performance')
tex_vex = np.reshape(textures, (512*512,num_textures), order='F').T
rand_vex = np.reshape(random, (512*512,num_textures), order='F').T
diff_mean = (np.mean(rand_vex[:n_samples,:], axis=0) - np.mean(tex_vex[:n_samples,:], axis=0))

test = np.concatenate((tex_vex[500:600,:], rand_vex[500:600,:]), axis=0)
y = np.ones((200,1))
y[:100]=-1
test = test[shuff,:]
y = y[shuff]
corr = 0

print('==> Training')
k_tex = fl.Train(textures[:,:,:n_samples], filter_size, step_size, out_dimension, LR, nonlinearity)
k_rand = fl.Train(random[:,:,:n_samples], filter_size, step_size, out_dimension, LR, nonlinearity)

k_tex = k_tex[:,:,0]
k_rand = k_rand[:,:,0]

k = np.multiply(0.5,k_tex+k_rand).T

#w = np.dot(k,diff_mean).T
w = np.multiply(k[:,0],diff_mean) # works because k is vector and 

for i in range(200):
    #x = np.reshape(test[:,:,i],262144, order='F') # 512*512
    x = test[i,:]
    yhat = np.sign(np.dot(w,x.T))
    if (yhat == y[i]):
        corr = corr+1

pc = corr/200.
print()
if (pc < 0.5):
    print('flipped')
    pc = 1.-pc
print('==> Percent Correct')
print(pc)





