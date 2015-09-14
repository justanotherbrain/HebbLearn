import sys
import scipy.ndimage
import os.path
import HebbLearn as hl
import numpy as np
import matplotlib.pyplot as plt

fl = hl.NonlinearGHA()
cat_a = '8' #monkey
cat_b = '10' #truck

def resize(data):
    tmp = np.reshape(data, (np.shape(data)[0],96,96,3), order='F')
    tmp = np.swapaxes(tmp,0,1)
    tmp = np.swapaxes(tmp,1,2)
    tmp = np.swapaxes(tmp,2,3)
    return tmp
    

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
    filter_size = 96
    step_size = 96 
    out_dimension = 1
    LR = 1
    n_samples = 500

nonlinearity = hl.DIVTANH
LR=1


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


#plt.imshow(output, cmap=plt.get_cmap('gray'))
#plt.show()


print('==> Training')
k_a = fl.Train(a_train[:,:,:n_train], filter_size, step_size, out_dimension, LR, nonlinearity)
k_b = fl.Train(b_train[:,:,:n_train], filter_size, step_size, out_dimension, LR, nonlinearity)

#a_pop = np.zeros((96,96))
#b_pop = np.zeros((96,96))
#for i in range(n_samples):
#    a_pop = a_pop + fl.ImageReconstruction(a_train[:,:,i], np.reshape(k_a,(out_dimension,96*96,1)), filter_size, step_size, nonlinearity)
#    b_pop = b_pop + fl.ImageReconstruction(b_train[:,:,i], np.reshape(k_b,(out_dimension,96*96,1)), filter_size, step_size, nonlinearity)

#apv = np.reshape(a_pop, (96*96,1), order='F').T/n_train
#bpv = np.reshape(b_pop, (96*96,1), order='F').T/n_train
#diff_mean = (np.mean(apv[:n_train,:], axis=0) - np.mean(bpv[:n_train,:], axis=0))



#k_a = k_a[:,:,0]
#k_b = k_b[:,:,0]
kdim = np.shape(k_a)
ka = np.zeros((kdim[0],kdim[1]*kdim[2])) #realign filters
kb = np.zeros((kdim[0],kdim[1]*kdim[2]))
for i in range(kdim[2]):
    for j in range(kdim[0]):
        ka[j,i*kdim[1]:(i+1)*kdim[1]]=k_a[j,:,i]
        kb[j,i*kdim[1]:(i+1)*kdim[1]]=k_b[j,:,i]


k = np.multiply(0.5,ka+kb).T

#w = np.dot(k[:,0],np.diag(diff_mean))
w = np.zeros((kdim[0], kdim[1]*kdim[2]))
for i in range(kdim[0]):
    w[i,:] = np.multiply(k[:,i],diff_mean) # works since both are vectors


print('')
print('')
print('==> Testing')
n_test = np.shape((test))[0]
yhs = np.zeros((np.shape(test)[0],1))
for i in range(n_test):
    #x = np.reshape(test[i,:],96*96, order='F') 
    x = test[i,:]
    #kx = np.reshape(fl.ImageReconstruction(np.reshape(x,(96,96),order='F'), np.reshape(k,(kdim[0],kdim[1],kdim[2])), filter_size, step_size, nonlinearity),96*96, order='F')
    #yhat = np.sign(np.dot(w,x.T))
    yhat = np.sign(np.sum(np.dot(w,x.T)))
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




