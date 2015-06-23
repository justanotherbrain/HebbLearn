import os
import math
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


# rgb2gray
# luminance preserving rgb2gray conversion
#
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])



# LoadImage
# Load an image as a numpy array
#
# file_name : directory of file to open
# returns : image as numpy array
def LoadImage(file_name):
    image = misc.imread(file_name)
    return image/255


# KD
# The Kronecker Delta Function - returns 1,
# if inputs are same, returns 0 otherwise
#
# m, n : input numbers to test
def KD(m, n):
    if (m==n):
        d=1;
    else:
        d=0;
    return d


# LinearGHA
# The Linear Generalized Hebbian Algorithm for
# a single image. There is no temporal component
# for this algorithm.
#
# in_vec : Nx1 dimensional numpy input vector
# weights : MxN dimensional numpy weight matrix
# out_vec : Mx1 dimensional numpy output vector
# LR : Learning rate, in (0,1)
# returns : new_weights updated weight matrix
def LinearGHA(in_vec, weights, out_vec, LR):
    N = in_vec.shape[0]
    M = out_vec.shape[0]
    
    # Check dimension compatibility
    if (weights[0] != M or weights[1] != N):
        raise ValueError('weight matrix dimensions and input or output vector dimensions do not align')
    
    if (in_vec.shape[1] > 1):
        raise ValueError('input vector is not an Nx1 dimensional vector')

    if (out_vec.shape[1] > 1):
        raise ValueError('output vector is not an Mx1 dimensional vector')

    # GHA from Sanger, 1989
    LT = np.tril(np.dot(out_vec, out_vec.T))
    new_weights = LR * (np.dot(out_vec, in_vec.T) - np.dot(LT,weights))
    return new_weights



# GetOutput_LinearGHA
# Get the output vector for the Linear GHA algorithm
#
# input_vec : Nx1 input vector
# weights : MxN weight matrix
# returns : Mx1 output vector
def GetOutput_LinearGHA(in_vec, weights):
    return numpy.dot(weights, input_vec)


# TrainLinearGHA
# Trains the LinearGHA over a certain number of iterations
#
# in_vec : Nx1 dimensional input vector
# weights : MxN weight matrix
# returns : weights (updated) and im_out - Nx1 dimensional output vector

def TrainLinearGHA(in_vec, weights, iterations):
    for t in range(1,iterations):
        out_vec = GetOutput_LinearGHA(in_vec, weights)
        LR = 1/t
        weights = LinearGHA(in_vec, weights, out_vec, LR)

    im_out = GetOutput_LinearGHA(in_vec, weights)
    return weights, im_out


# InitializeWeights
# Initialize the weight matrix (random)
#
# in_vec : input vector Nx1 dimensional
# out_dimension : dimension of output vector (M)
def InitializeWeights(in_vec, out_dimension):
    if (in_vec.shape[1]>1):
        raise ValueError('input vector is not Nx1 dimensional')
    return np.random.rand((in_vec.shape[0], out_dimension))



def Demo_LinearGHA():
    goat = rgb2gray(LoadImage('goat.jpg'))
    plt.imshow(g, cmap=plt.get_cmap('gray'))
    plt.show()

    N = goat.shape[0]*goat.shape[1]
    M = 100
    Iterations = 1000

    in_vec = np.reshape(goat, (N,1))
    weights = InitializeWeights(in_vec, M)
    
    TrainGHA(in_vec, weights, Iterations)






