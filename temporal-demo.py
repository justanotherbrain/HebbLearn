import sys
import HebbLearn as hl
import numpy as np
import matplotlib.pyplot as plt
import cv2

# load video as numpy array
fn = 'short.avi'
vid = hl.LoadVideo(fn) 

TemporalGHA = hl.TemporalGHA()

# filter size has to be odd!
if len(sys.argv)>1:
    filter_size = int(sys.argv[1])
    out_dimension = int(sys.argv[3])
    LR = float(sys.argv[4])
else:
    filter_size = 9
    out_dimension = 9
    LR = 0.000001

time_filter = TemporalGHA.GetTimeFilter(filter_size)

weights = TemporalGHA.Train(vid, filter_size, out_dimension, LR, time_filter)

output = TemporalGHA.VideoReconstruction(vid, weights, filter_size, time_filter)
out_vid = hl.GenerateVideo(output, 'video_out.avi')

hl.DisplayFrames(output)
