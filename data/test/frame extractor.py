# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:01:55 2018

@author: 박진우
"""

import cv2
vidcap = cv2.VideoCapture('./hoi_01.mp4')
success,image = vidcap.read()
count = 0
while success:
  # cv2.imwrite("frame%04d.png" % count, image)     # save frame as JPEG file   
  
  # Image Rotate

  image = cv2.transpose(image) 
  image = cv2.flip(image, 1)
  cv2.imwrite('./inputs/'+str(count) + ".jpg", image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1