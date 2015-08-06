import os
import math
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
try:
	import cv2
except:
	print('cv2 not available')
	pass


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

# Load Video
# Load a video as a numpy array
#
# file_name : directory of video file to open
# returns : tensor of image (b&w)
def LoadVideo(fn):
    cap = cv2.VideoCapture(fn)
    num_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_rows = np.shape(gray)[0]
    num_cols = np.shape(gray)[1]
    
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0)

    vid = np.zeros((num_rows, num_cols, num_frames))
    
    for f in range(num_frames):
        ret, frame = cap.read()
        vid[:,:,f] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cap.release()

    return vid

# GenerateVideo
# 
# Generates a video from a grayscale numpy array and saves to file
def GenerateVideo(raw, fn):
    num_rows = np.shape(raw)[0]
    num_cols = np.shape(raw)[1]
    num_frames = np.shape(raw)[2]

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter(fn, 1196444237, 20, (num_rows, num_cols))
    
    for f in range(num_frames):
        #can't use normalized image for write
        processed = (raw[:,:,f]*255.0).astype('uint8')
        processed = np.repeat(processed,3,axis=1)
        processed = processed.reshape(num_rows,num_cols,3)
        out.write(processed)
    
    out.release()


def DisplayFrames(video):
    num_rows = np.shape(video)[0]
    num_cols = np.shape(video)[1]
    num_frames = np.shape(video)[2]
    
    f = 0
    plt.imshow(video[:,:,f], cmap=plt.get_cmap('gray'))
    plt.show(block=False)
    while(True):
        f = 0
        print("\033[A                                           \033[A")
        x = input("Press f: forward, b: back, q: quit  :  ")
        if (x == "f"):
            if ((f+1) < num_frames):
                f = f+1
        elif (x == "b"):
            if ((f-1) >= 0):
                f = f-1
        elif (x == "q"):
            break
        else:
            f = f

        plt.imshow(video[:,:,f], cmap=plt.get_cmap('gray'))
        plt.show(block=False)

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



# FixedLinearGHA
#
# This class computes a linear generalized hebb 
# algorithm - the weight matrix contains
# the weights for each center-surround.
class FixedLinearGHA():
    def LinearGHA(self, in_vec, weights, out_vec, LR):
        if (np.max(in_vec) ==  0):
            in_vec = in_vec+.0000001
        if (np.max(out_vec) == 0):
            out_vec = out_vec + 0.0000001
        LT = np.tril(np.dot(out_vec, out_vec.T))
        new_weights = weights + LR * (np.dot(out_vec, in_vec.T) - np.dot(LT,weights))
        return new_weights/np.max(new_weights)


    def ResizeImage(self, im, filter_size):
        return im[0:(im.shape[0]-(im.shape[0]%filter_size)),0:(im.shape[1]-(im.shape[0]%filter_size))]


    def InitializeWeights(self, im, filter_size, out_dimension):
        num_rows = im.shape[0]/filter_size
        num_cols = im.shape[1]/filter_size
        return np.random.rand(out_dimension,(filter_size*filter_size),num_rows*num_cols)


    def Train(self, raw_data, filter_size, out_dimension, LR):
        sample_size = np.shape(data)[2]
        # crop video to correct dimensions
        temp = self.ResizeImage(raw_data[:,:,0], filter_size)
        data = np.zeros((np.shape(temp)[0],np.shape(temp)[1],sample_size))
        for t in range(num_frames):
            data[:,:,t] = self.ResizeImage(raw_data[:,:,t], filter_size)

        num_rows = data.shape[0]/filter_size
        num_cols = data.shape[1]/filter_size
        weights = self.InitializeWeights(data[:,:,0], filter_size, out_dimension)
        for f in range(filter_size, sample_size):
            w = 0
            for c in range(num_cols-1):
                for r in range(num_rows-1):
                    row_start = r*filter_size
                    row_end = (r+1)*filter_size
                    col_start = c*filter_size
                    col_end = (c+1)*filter_size

                    img = np.zeros((filter_size, filter_size))
                    in_vec = np.reshape(img,(filter_size*filter_size,1))
                    out_vec = self.GetOutput(in_vec, weights[:,:,w])
                    weights[:,:,w] = self.LinearGHA(in_vec, weights[:,:,w], out_vec, LR)
                    w = w+1
        return weights



    def GetOutput(self, in_vec, weights):
        return np.dot(weights, in_vec)





# HierarchicalGHA
#
#
class HierarchicalGHA():
    


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
        if (np.max(in_vec) ==  0):
            in_vec = in_vec+.0000001
        if (np.max(out_vec) == 0):
            out_vec = out_vec + 0.0000001
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
                    
                    frame = np.zeros((filter_size, filter_size))
                    for t in range(np.shape(time_filter)[2]):
                        frame = frame + np.multiply(vid[row_start:row_end, col_start:col_end, f-t], time_filter[:,:,t])
                    in_vec = np.reshape(frame,(filter_size*filter_size,1))
                    out_vec = self.GetOutput(in_vec, weights[:,:,w])
                    weights[:,:,w] = self.LinearGHA(in_vec, weights[:,:,w], out_vec, LR)
                    w = w+1
        return weights
   


    def GetOutput(self, in_vec, weights):
        return np.dot(weights, in_vec)

    
    
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
        
        num_rows = temp.shape[0]/filter_size
        num_cols = temp.shape[1]/filter_size
        out_dimension = weights.shape[0]
        output = np.zeros((temp.shape[0],temp.shape[1],num_frames))
        for f in range(filter_size, num_frames):
            w = 0
            for c in range(num_cols):
                for r in range(num_rows):
                    row_start = r*filter_size
                    row_end = (r+1)*filter_size
                    col_start = c*filter_size
                    col_end = (c+1)*filter_size
                    
                    frame = np.zeros((filter_size,filter_size))
                    for t in range(np.shape(time_filter)[2]):
                        frame = frame + np.multiply(vid[row_start:row_end, col_start:col_end, f-t], time_filter[:,:,t])

                    in_vec = np.reshape(frame,(filter_size*filter_size,1))
                    out_vec = np.dot(weights[:,:,w].T, np.dot(weights[:,:,w], in_vec))
                    output[row_start:row_end,col_start:col_end,f] = np.reshape(out_vec,(filter_size,filter_size))
                    w = w + 1
        return output

