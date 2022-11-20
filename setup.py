# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:30:32 2022

@author: Andres Fandos

Script to create the necessary folders
"""

import os
    

if __name__ == "__main__":

    os.mkdir('data')
    os.mkdir('data/images')
    os.mkdir('data/images/face')
    os.mkdir('data/images/noface')
    
    os.mkdir('data/images_resized')
    os.mkdir('data/labels')
    
    os.mkdir('aug_data')
    os.mkdir('aug_data/images')
    os.mkdir('aug_data/labels')