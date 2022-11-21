# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:11:55 2022

@author: Andres Fandos

The script captures images from the camera to train the model.
If you already have images ignore this script and copy them
directly to the "images" folder.
"""

import os
import cv2
import time
import configargparse


if __name__ == "__main__":
    
    p = configargparse.ArgumentParser()
    p.add_argument('--images_to_take', type=int, default=100, help='Number of images to take from the camera')
    opt = p.parse_args()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("Could not open video device")
    
    for imgnum in range(opt.images_to_take):
        
        print('Collecting image {}'.format(imgnum))
        
        ret, frame = cap.read()
        
        imgname = os.path.join('data/images',f'{str(imgnum)}.jpg')
        
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        
        time.sleep(0.5)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
