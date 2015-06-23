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
    return misc.imread(file_name)


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
	LT = np.tril(np.dot(out_vec, out_vec.T))
	new_weights = weights + LR * (np.dot(out_vec, in_vec.T) - np.dot(LT,weights))
	return new_weights



# GetOutput_LinearGHA
# Get the output vector for the Linear GHA algorithm
#
# input_vec : Nx1 input vector
# weights : MxN weight matrix
# returns : Mx1 output vector
def GetOutput_LinearGHA(in_vec, weights):
    return np.dot(weights, in_vec)


# TrainLinearGHA
# Trains the LinearGHA over a certain number of iterations
#
# in_vec : Nx1 dimensional input vector
# weights : MxN weight matrix
# returns : weights (updated) and im_out - Nx1 dimensional output vector
#
#def TrainLinearGHA(in_vec, weights, iterations):
#    for t in range(1,iterations):
#        out_vec = GetOutput_LinearGHA(in_vec, weights)
#        LR = 1/t
#        weights = LinearGHA(in_vec, weights, out_vec, LR)
#
#    im_out = GetOutput_LinearGHA(in_vec, weights)
#    return weights, im_out


# ResizeImage
# Resize image by cropping out right and bottom rows until a full convolution
# is possible
#
# im : input image PxQ, grayscale
# filter_size : size of filter to be used
# returns : P-P%filter_size x Q-Q%filter_size image
def ResizeImage(im, filter_size):
	return im[0:(im.shape[0]-(im.shape[0]%filter_size)),0:(im.shape[1]-(im.shape[0]%filter_size))]



# TrainSlidingLinearGHA
# Trains the LinearGHA over a certain number of iterations. Take an
# input image and splits into non-overlapping segments. Trains a
# weight matrix over the entire image, by sliding filter image. 
#
# input_image : PxQ, grayscale input image
# filter_size : dimension of filter
# out_dimension : size of the output vector
# iterations : number of iterations over input_image
def TrainSlidingLinearGHA(input_image, filter_size, out_dimension, iterations):
	im = ResizeImage(input_image, filter_size)
	num_rows = im.shape[0]/filter_size
	num_cols = im.shape[1]/filter_size
	weights = InitializeWeights(filter_size, out_dimension)
	t = 1
	for i in range(iterations):
		for c in range(num_cols-1):
			for r in range(num_rows-1):
				row_start = r*filter_size
				row_end = (r+1)*filter_size
				col_start = c*filter_size
				col_end = (c+1)*filter_size
				
				frame = im[row_start:row_end, col_start:col_end]
				in_vec = np.reshape(frame,(filter_size*filter_size,1))
				out_vec = GetOutput_LinearGHA(in_vec, weights)
				LR = 1/t
				LR = 0.01 # empirical fix point for now
				weights = LinearGHA(in_vec, weights, out_vec, LR)
				t=t+1
	return weights



# ReconstructSlidingLinearGHA
# Reconstruct an output image by passing input image through weights
def ReconstructSlidingLinearGHA(input_image, weights, filter_size):
	im = ResizeImage(input_image, filter_size)
	num_rows = im.shape[0]/filter_size
	num_cols = im.shape[1]/filter_size
	out_dimension = weights.shape[0]
	output = np.zeros((im.shape[0],im.shape[1]))
	for c in range(num_cols):
		for r in range(num_rows):
			row_start = r*filter_size
			row_end = (r+1)*filter_size
			col_start = c*filter_size
			col_end = (c+1)*filter_size
			
			frame = im[row_start:row_end, col_start:col_end]
			in_vec = np.reshape(frame,(filter_size*filter_size,1))
			out_vec = np.zeros(in_vec.T.shape)
			for o in range(out_dimension):
				# element-wise multiplication
				out_vec = out_vec + (weights[o,:]*in_vec.T)/out_dimension
			output[row_start:row_end,col_start:col_end] = np.reshape(out_vec,(filter_size,filter_size))
	return output


# TrainFixedLinearGHA
# Trains the LinearGHA over multiple images for certain number of
# iterations. There are non-overlapping filters that segment the 
# entire image - each learns weights for a specific segment of the
# image space.
#
# input_images : PxQxT, grayscale tensor of T images
# filter_size : dimension of filter
# iterations : number of iterations to repeat training over entire sample set
def TrainFixedLinearGHA(input_images, filter_size, iterations):
	return 0
# InitializeWeights
# Initialize the weight matrix (random)
#
# filter_size : size of filter to be used
# out_dimension : dimension of output vector (M)
def InitializeWeights(filter_size, out_dimension):
    return np.random.rand(out_dimension,(filter_size*filter_size))



def Demo_LinearGHA():
	goat = rgb2gray(LoadImage('goat.jpg'))
	if np.max(goat)>1:
		goat = goat/255
	
	plt.imshow(goat, cmap=plt.get_cmap('gray'))
	plt.show()
	
	filter_size = 8
	iterations = 2
	out_dimension = 8
	
	weights = TrainSlidingLinearGHA(goat, filter_size, out_dimension, iterations) 
	
	output = ReconstructSlidingLinearGHA(goat, weights, filter_size)
	plt.imshow(output, cmap=plt.get_cmap('gray'))
	plt.show()
	
	print(weights)


