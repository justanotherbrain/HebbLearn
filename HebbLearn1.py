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



class SlidingLinearGHA():
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
    #
    def LinearGHA(self, in_vec, weights, out_vec, LR):
        LT = np.tril(np.dot(out_vec, out_vec.T))
        new_weights = weights + LR * (np.dot(out_vec, in_vec.T) - np.dot(LT,weights))
        return new_weights/np.max(new_weights)
    
    
    
    # GetOutput_LinearGHA
    # Get the output vector for the Linear GHA algorithm
    #
    # input_vec : Nx1 input vector
    # weights : MxN weight matrix
    # returns : Mx1 output vector
    def GetOutput(self, in_vec, weights):
        return np.dot(weights, in_vec)
    
    
    # ResizeImage
    # Resize image by cropping out right and bottom rows until a full convolution
    # is possible
    #
    # im : input image PxQ, grayscale
    # filter_size : size of filter to be used
    # returns : P-P%filter_size x Q-Q%filter_size image
    def ResizeImage(self, im, filter_size):
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
    def Train(self, input_image, filter_size, out_dimension, LR, iterations):
        im = self.ResizeImage(input_image, filter_size)
        num_rows = im.shape[0]/filter_size
        num_cols = im.shape[1]/filter_size
        weights = self.InitializeWeights(filter_size, out_dimension)
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
                    out_vec = self.GetOutput(in_vec, weights)
                    weights = self.LinearGHA(in_vec, weights, out_vec, LR)
                    t=t+1
        return weights



    # SlidingMaskLinearGHA
    # Pass input image through weights as if weights are mask
    def Mask(self, input_image, weights, filter_size):
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


    # SlidingImageReconstruction
    # Reconstruct image by passing through filter and reconstructing
    def ImageReconstruction(self, input_image, weights, filter_size):
        im = self.ResizeImage(input_image, filter_size)
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
                out_vec = np.dot(weights.T, np.dot(weights, in_vec))
                output[row_start:row_end,col_start:col_end] = np.reshape(out_vec,(filter_size,filter_size))
        return output


    # InitializeWeights
    # Initialize the weight matrix (random)
    #
    # filter_size : size of filter to be used
    # out_dimension : dimension of output vector (M)
    def InitializeWeights(self, filter_size, out_dimension):
        return np.random.rand(out_dimension,(filter_size*filter_size))



    # VisualizeFilter
    def VisualizeFilter(self, weights):
        fs = np.sqrt(weights.shape[1])
        for i in range(weights.shape[0]):
            f = np.reshape(weights[i,:], (fs,fs))
            plt.imshow(f, cmap=plt.get_cmap('gray'),interpolation='none')
            plt.colorbar()
            plt.show()




# The TemporalGHA Algorithm
# This module is for training on video in order to predict frames using
# a simple Hebbian update rule.
#
# This algorithm saves a tensor of non-overlapping weight matrices that 
# correspond to an area of overlap
class TemporalGHA:
    def LinearGHA(self, in_vec, weights, out_vec, LR):
        LT = np.tril(np.dot(out_vec, out_vec.T))
        new_weights = weights + LR * (np.dot(out_vec, in_vec.T) - np.dot(LT,weights))
        return new_weights/np.max(new_weights)


    def ResizeFrame(self, im, filter_size):
        return im[0:(im.shape[0]-(im.shape[0]%filter_size)),0:(im.shape[1]-(im.shape[0]%filter_size))]


    def InitializeWeights(self, im, filter_size, out_dimension):
        num_rows = im.shape[0]/filter_size
        num_cols = im.shape[1]/filter_size
        return np.random.rand(out_dimension,(filter_size*filter_size),num_rows*num_cols)
    
    
    def Train(self, input_video, filter_size, out_dimension, LR, time_filter):
        num_frames = np.shape(input_video)[2] 
        # crop video to correct dimensions
        temp = self.ResizeFrame(input_video[:,:,0], filter_size)
        vid = np.zeros((np.shape(temp)[0],np.shape(temp)[1],num_frames))
        for t in range(num_frames):
            vid[:,:,t] = self.ResizeFrame(input_video[:,:,t], filter_size)

        num_rows = vid.shape[0]/filter_size
        num_cols = vid.shape[1]/filter_size
        weights = self.InitializeWeights(vid[:,:,0], filter_size, out_dimension)
        for f in range(filter_size, num_frames):
            w = 0
            for c in range(num_cols-1):
                for r in range(num_rows-1):
                    row_start = r*filter_size
                    row_end = (r+1)*filter_size
                    col_start = c*filter_size
                    col_end = (c+1)*filter_size
                    
                    frame = np.zeros(filter_size, filter_size)
                    for t in range(filter_size):
                        frame = frame + np.multiply(vid[row_start:row_end, col_start:col_end, f-t], time_filter[:,:,t])
                    in_vec = np.reshape(frame,(filter_size*filter_size,1))
                    out_vec = self.GetOutput(in_vec, weights)
                    weights[:,:,w] = self.LinearGHA(in_vec, weights, out_vec, LR)
                    w = w+1
        return weights
    
    
    
    
    def GetTimeFilter(self, filter_size):
        n = filter_size
        time_filter = np.zeros((n, n, np.ceil(n/2)+1))
        for t in range(int(np.ceil(n/2))+1):
            if (t==0):
                time_filter[np.floor(n/2),np.floor(n/2),t] = 1
            else:
                o = int(np.floor(n/2) - t) # origin
                c = int(o + (t*2)) # ceiling
                for i in range(o,c+1):
                    time_filter[o,i,t] = 1
                    time_filter[c,i,t] = 1
                    time_filter[i,o,t] = 1
                    time_filter[i,c,t] = 1
        return time_filter

    def VideoReconstruction(self, input_video, weights, filter_size, time_filter):
        num_frames = np.shape(input_video)[2]
        # crop video to correct dimensions
        temp = self.ResizeFrame(input_video[:,:,0], filter_size)
        vid = np.zeros((np.shape(temp)[0],np.shape(temp)[1],num_frames))
        for t in range(num_frames):
            vid[:,:,t] = self.ResizeFrame(input_video[:,:,t], filter_size)
        
        num_rows = im.shape[0]/filter_size
        num_cols = im.shape[1]/filter_size
        out_dimension = weights.shape[0]
        output = np.zeros((temp.shape[0],temp.shape[1],num_frames))
        for f in range(filter_size, num_frame):
            w = 0
            for c in range(num_cols):
                for r in range(num_rows):
                    row_start = r*filter_size
                    row_end = (r+1)*filter_size
                    col_start = c*filter_size
                    col_end = (c+1)*filter_size
                    
                    frame = np.zeros(filter_size,filter_size)
                    for t in range(filter_size):
                        frame = frame + np.multiply(vid[row_start:row_end, col_start:col_end, f-t], time_filter[:,:,t])

                    in_vec = np.reshape(frame,(filter_size*filter_size,1))
                    out_vec = np.dot(weights.T, np.dot(weights[:,:,w], in_vec))
                    output[row_start:row_end,col_start:col_end,f] = np.reshape(out_vec,(filter_size,filter_size))
                    w = w + 1
        return output

