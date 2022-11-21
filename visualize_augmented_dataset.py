# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 18:15:11 2022

@author: Andres Fandos

Script to display the images in the augmented dataset with their corresponding
bounding boxes around the face in the image.
"""

import os
import cv2
import json
import random
import numpy as np


if __name__ == "__main__":
    
    images_path = os.path.join('aug_data', 'labels')
    images = os.listdir(os.path.join(images_path))
    random.shuffle(images)
    coords = [0, 0, 0, 0]
    
    for image in images:
    
        with open(os.path.join(images_path, image), 'r') as f:
            label = json.load(f)
        
        img = cv2.imread(os.path.join('aug_data', 'images',
                                      image.split('.')[0] + '.' + image.split('.')[1] + '.jpg'))
        
        coords[0] = label["bbox"][0]
        coords[1] = label["bbox"][1]
        coords[2] = label["bbox"][2]
        coords[3] = label["bbox"][3]
        
        cv2.rectangle(img=img,
                      pt1=tuple(np.multiply(np.array(coords[:2]), [img.shape[1], img.shape[0]]).astype(int)),
                      pt2=tuple(np.multiply(np.array(coords[2:]), [img.shape[1], img.shape[0]]).astype(int)),
                      color=(0, 0, 255),
                      thickness=2)
    
        cv2.imshow('Image', img)
        key = cv2.waitKey(1000)
        
        # Si se presiona la tecla ESC se sale del bucle
        if key == 27:
            break
        
    cv2.destroyAllWindows()
