# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 18:13:17 2018

@author: 박진우
"""

import cv2
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    print(files)
    #for sorting the file names properly
#    files.sort(key = lambda x: int(x[5:-4]))
    # files.sort(key = lambda x: int(x[0:-4]))
    files.sort(key = lambda x: int(x.split('.')[0]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def main():
    pathIn= './outputs/'
    pathOut = 'new1.avi'
    fps = 15.0   #0.5배속
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()