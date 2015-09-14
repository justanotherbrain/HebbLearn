import sys
import scipy.ndimage
import os.path
import HebbLearn as hl
import numpy as np
import matplotlib.pyplot as plt

fl = hl.NonlinearGHA()
cat_a = '1' #monkey
cat_b = '2' #truck

def resize(data):
    tmp = np.reshape(data, (np.shape(data)[0],96,96,3), order='F')
    tmp = np.swapaxes(tmp,0,1)
    tmp = np.swapaxes(tmp,1,2)
    tmp = np.swapaxes(tmp,2,3)
    return tmp


def realign(image, filter_size, step_size):
    im = fl.ResizeImage(image, filter_size)
    #realigned = np.zeros((im.shape[0]*im.shape[1],1))
    row_steps = ((im.shape[0] - filter_size)/step_size)+1
    col_steps = ((im.shape[1] - filter_size)/step_size)+1
    realigned = np.zeros((row_steps*filter_size*col_steps*filter_size,1))
    s = 0
    fs2 = filter_size*filter_size
    for c in range(col_steps):
        for r in range(row_steps):
            rs = r*step_size
            re = r*step_size + filter_size
            cs = c*step_size
            ce = c*step_size + filter_size

            patch = im[rs:re, cs:ce]
            map = np.reshape(patch,(fs2,1))
            realigned[s*fs2:(s+1)*fs2] = map
    return realigned


if os.path.isfile('stl-data/c1_train.npy'):
    print('==> Load previously saved textures data')
    tmp_a_train = resize(np.load('stl-data/c'+cat_a+'_train.npy'))/255.
    tmp_b_train = resize(np.load('stl-data/c'+cat_b+'_train.npy'))/255.
    
    tmp_a_test = resize(np.load('stl-data/c'+cat_a+'_test.npy'))/255.
    tmp_b_test = resize(np.load('stl-data/c'+cat_b+'_test.npy'))/255.

n_train = np.shape(tmp_a_train)[3]
n_test = np.shape(tmp_a_test)[3]

print('==> preprocessing data')
a_train = np.zeros((96,96,n_train))
b_train = np.zeros((96,96,n_train))
a_test = np.zeros((96,96,n_test))
b_test = np.zeros((96,96,n_test))


for i in range(n_train):
    a_train[:,:,i] = hl.rgb2gray(tmp_a_train[:,:,:,i])
    b_train[:,:,i] = hl.rgb2gray(tmp_b_train[:,:,:,i])

for i in range(n_test):
    a_test[:,:,i] = hl.rgb2gray(tmp_a_test[:,:,:,i])
    b_test[:,:,i] = hl.rgb2gray(tmp_b_test[:,:,:,i])

print('==> mean centering data')

pop_mean = np.mean(np.concatenate((a_train,b_train),axis=2))
a_train = a_train - pop_mean
b_train = b_train - pop_mean
a_test = a_test - pop_mean
b_test = b_test - pop_mean

pop_std = np.std(np.concatenate((a_train,b_train),axis=2))
a_train = a_train/pop_std
b_train = b_train/pop_std
a_test = a_test/pop_std
b_test = b_test/pop_std


if len(sys.argv)>1:
    filter_size = int(sys.argv[1])
    step_size = int(sys.argv[2])
    out_dimension = int(sys.argv[3])
    LR = float(sys.argv[4])
    n_samples = int(sys.argv[5])
else:
    filter_size = 4
    step_size = 2
    out_dimension = 1
    LR = 1
    n_samples = 500

nonlinearity = hl.TANH
#LR=.000000001
LR=1
#LR=.5

print('==> Classification performance')
a_vex = np.reshape(np.concatenate((a_train,a_test),axis=2), (96*96,n_train+n_test), order='F').T
b_vex = np.reshape(np.concatenate((b_train,b_test),axis=2), (96*96,n_train+n_test), order='F').T
diff_mean = (np.mean(a_vex[:n_train,:], axis=0) - np.mean(b_vex[:n_train,:], axis=0))

test = np.concatenate((a_vex[n_train:], b_vex[n_train:]), axis=0)
y = np.ones((np.shape(test)[0],1))
y[n_test:]=-1
shuff = np.random.permutation(np.shape(test)[0])
test = test[shuff,:]
y = y[shuff]
corr = 0


print('==> Training')
k_a = fl.Train(a_train[:,:,:n_train], filter_size, step_size, out_dimension, LR, nonlinearity)
k_b = fl.Train(b_train[:,:,:n_train], filter_size, step_size, out_dimension, LR, nonlinearity)

#k_a = k_a[:,:,0]
#k_b = k_b[:,:,0]
kdim = np.shape(k_a)
ka = np.zeros((kdim[0],kdim[1]*kdim[2])) #realign filters
kb = np.zeros((kdim[0],kdim[1]*kdim[2]))
for i in range(kdim[2]):
    for j in range(kdim[0]):
        ka[j,i*kdim[1]:(i+1)*kdim[1]]=k_a[j,:,i]
        kb[j,i*kdim[1]:(i+1)*kdim[1]]=k_b[j,:,i]

ma = np.zeros((kdim[0],kdim[1]*kdim[2]))
mb = np.zeros((kdim[0],kdim[1]*kdim[2]))
for j in range(kdim[0]):
    for i in range(n_train):
        ma[j,:] = ma[j,:] + np.multiply(ka[j,0], np.tanh(realign(np.reshape(a_vex[i,:], (96,96)), filter_size, step_size).T))
        mb[j,:] = mb[j,:] + np.multiply(kb[j,0], np.tanh(realign(np.reshape(b_vex[i,:], (96,96)), filter_size, step_size).T))

diff_mean = ma-mb

k = np.multiply(0.5,ka+kb).T

#w = np.dot(k[:,0],np.diag(diff_mean))
w = np.zeros((kdim[0], kdim[1]*kdim[2]))
for i in range(kdim[0]):
    w[i,:] = np.multiply(k[:,i],diff_mean[i,:]) # works since both are vectors

#w = np.multiply(k[:,0],diff_mean)
print('')
print('')
print('==> Testing')
n_test = np.shape((test))[0]
yhs = np.zeros((np.shape(test)[0],1))
for i in range(n_test):
    x = realign(np.reshape(test[i,:],(96,96)), filter_size, step_size)
    yhat = np.sign(np.dot(w,x))
    #yhat = np.sign(np.sum(np.dot(w,x.T)))
    if (yhat == y[i]):
        corr = corr+1.
    pt = ((i+.0)/n_test)*100
    sys.stdout.write("\rTesting is %f percent complete" % pt)
    sys.stdout.flush()

pc = corr/np.shape(test)[0]
print('')
print('')
if (pc < 0.5):
    print('flipped')
    pc = 1.0-pc
print('==> Percent Correct')
print(pc)




