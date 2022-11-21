# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:49:28 2022

@author: Andres Fandos

This script performs a data augmentation increasing the number of images
by 60 using the albumentations library. The new images are stored in the
aug_data folder and their size is (160, 90). Different subwindows of
the images are taken.
"""

import os
import cv2
import json
import numpy as np
import configargparse
import albumentations as alb


if __name__ == "__main__":
    
    p = configargparse.ArgumentParser()
    p.add_argument('--width', type=int, default=256, help='Width of the subimage')
    p.add_argument('--height', type=int, default=144, help='Width of the subimage')
    p.add_argument('--num_subimages', type=int, default=100, help='Number of subimages to take')
    opt = p.parse_args()
    
    augmentor = alb.Compose([alb.RandomCrop(width=opt.width, height=opt.height), 
                             alb.HorizontalFlip(p=0.5), 
                             alb.RandomBrightnessContrast(p=0.2),
                             alb.RandomGamma(p=0.2), 
                             alb.RGBShift(p=0.2), 
                             alb.VerticalFlip(p=0.5)], 
                             bbox_params=alb.BboxParams(format='albumentations',
                                                        label_fields=['class_labels']))
    
    images_resized = os.listdir(os.path.join('data', 'images_resized'))
    
    for image in images_resized:
        
        img = cv2.imread(os.path.join('data', 'images_resized', image))

        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join('data', 'labels', f'{image.split(".")[0]}.json')
        
        if os.path.exists(label_path):
            
            with open(label_path, 'r') as f:
                label = json.load(f)

            x0 = label['shapes'][0]['points'][0][0]
            y0 = label['shapes'][0]['points'][0][1]
            x1 = label['shapes'][0]['points'][1][0]
            y1 = label['shapes'][0]['points'][1][1]
            
            if x0 < x1 and y0 < y1:
                coords[0] = x0
                coords[1] = y0
                coords[2] = x1
                coords[3] = y1
            elif x0 < x1 and y0 > y1:
                coords[0] = x0
                coords[1] = y1
                coords[2] = x1
                coords[3] = y0
            elif x0 > x1 and y0 < y1:
                coords[0] = x1
                coords[1] = y0
                coords[2] = x0
                coords[3] = y1
            elif x0 > x1 and y0 > y1:
                coords[0] = x1
                coords[1] = y1
                coords[2] = x0
                coords[3] = y0
              
            coords = list(np.divide(coords, [640, 360, 640, 360]))
            
        try:
            
            for x in range(opt.num_subimages):
                
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data',
                                         'images', f'{image.split(".")[0]}.{x}.jpg'),
                                         augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0 

                with open(os.path.join('aug_data', 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)
