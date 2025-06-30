# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 17:33:20 2022

@author: Andres Fandos

The script takes the images from the "images" folder and resizes them to a
resolution of (640, 360) so they are the same size.
"""

import os
import cv2
import configargparse


if __name__ == "__main__":
    
    p = configargparse.ArgumentParser()
    p.add_argument('--width', type=int, default=640, help='Width of the resized output image')
    p.add_argument('--height', type=int, default=360, help='Height of the resized output image')
    opt = p.parse_args()
    
    images_path = os.path.join('data', 'images')
        
    for image in os.listdir(images_path):
        
        img = cv2.imread(os.path.join(images_path, image))
        img_resized = cv2.resize(img, (opt.width, opt.height))
        cv2.imwrite(os.path.join('data', 'images_resized', image), img_resized)
