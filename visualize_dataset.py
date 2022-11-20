# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 18:15:11 2022

@author: Andres Fandos

Script to display the images in the dataset with their corresponding
bounding boxes around the face in the image.
"""

import os
import cv2
import json
import numpy as np


if __name__ == "__main__":
    
    images = os.listdir(os.path.join('data', 'labels'))
    
    for image in images: 
        
        img = cv2.imread(os.path.join('data', 'images_resized', image.split('.')[0] + '.jpg'))
        
        with open(os.path.join('data', 'labels', image), 'r') as f:
            label = json.load(f)
        
        coords = [0, 0, 0, 0]
        coords[0] = label['shapes'][0]['points'][0][0]
        coords[1] = label['shapes'][0]['points'][0][1]
        coords[2] = label['shapes'][0]['points'][1][0]
        coords[3] = label['shapes'][0]['points'][1][1]
        
        cv2.rectangle(img=img,
                      pt1=tuple(np.array(coords[:2]).astype(int)),
                      pt2=tuple(np.array(coords[2:]).astype(int)),
                      color=(0, 0, 255),
                      thickness=2)
    
        cv2.imshow('Image', img)
        key = cv2.waitKey(1000)
        
        # Si se presiona la tecla ESC se sale del bucle
        if key == 27:
            break
        
    cv2.destroyAllWindows()