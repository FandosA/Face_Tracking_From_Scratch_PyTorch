# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:30:32 2022

@author: Andres Fandos

Script to create the necessary folders
"""

import os
    

if __name__ == "__main__":

    os.mkdir('data')
    os.mkdir(os.path.join('data', 'images'))
    os.mkdir(os.path.join('data', 'images_resized'))
    os.mkdir(os.path.join('data', 'labels'))
    
    os.mkdir('aug_data')
    os.mkdir(os.path.join('aug_data', 'images'))
    os.mkdir(os.path.join('aug_data', 'labels'))
