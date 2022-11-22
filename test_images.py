# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:23:32 2022

@author: Andres Fandos

Script to test the model trained with images
"""

import os
import cv2
import torch
import random
import numpy as np
import configargparse
from vgg16 import VGG16
from torchvision.transforms import transforms


def plot_prediction():
    
    if label_pred.item() > opt.accuracy:
        
        coords[0] = bbox_coords_pred[0][0].item()
        coords[1] = bbox_coords_pred[0][1].item()
        coords[2] = bbox_coords_pred[0][2].item()
        coords[3] = bbox_coords_pred[0][3].item()
        
        cv2.putText(image, 'Face',
                    tuple(np.multiply(np.array(coords[:2]), [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        cv2.rectangle(img=image,
                      pt1=tuple(np.multiply(np.array(coords[:2]), [image.shape[1], image.shape[0]]).astype(int)),
                      pt2=tuple(np.multiply(np.array(coords[2:]), [image.shape[1], image.shape[0]]).astype(int)),
                      color=(0, 0, 255),
                      thickness=2)

    cv2.imshow('Image', image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_test_path', type=str, default='data/images_resized', help='Dataset path')
    p.add_argument('--log_dir', type=str, default='face_tracking_model', help='Name of the folder to load the model')
    p.add_argument('--checkpoint', type=str, default='checkpoint_43_best.pth', help='Checkpoint path')
    p.add_argument('--accuracy', type=float, default=0.75, help='Accuracy threshold')
    p.add_argument('--device', type=str, default='gpu', help='Choose the device: "gpu" or "cpu"')
    opt = p.parse_args()
    
    assert os.path.isdir(opt.log_dir), 'The folder log_dir does not exists'
    
    
    # Select device
    if opt.device == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Device assigned: GPU (' + torch.cuda.get_device_name(device) + ')\n')
    else:
        device = torch.device("cpu")
        if not torch.cuda.is_available() and opt.device == 'gpu':
            print('GPU not available, device assigned: CPU\n')
        else:
            print('Device assigned: CPU\n')
    
    # Load the model and create the model
    model = VGG16(in_channels=3, out_channels_bbox=4, out_channels_label=1)
    state_dict = torch.load(os.path.join(opt.log_dir, "checkpoints", opt.checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    images_test_path = os.path.join(opt.dataset_test_path)
    transformer = transforms.ToTensor()
    images = os.listdir(images_test_path)
    random.shuffle(images)
    coords = [0, 0, 0, 0]
    
    # Make predictions over the test images and plot them
    for image_test in images:
        
        image = cv2.imread(os.path.join(images_test_path, image_test))
        img_resized = cv2.resize(image, (256, 144))
        image_tensor = transformer(img_resized)
        image_tensor = image_tensor.to(device)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        
        # Predict if there's face in the image and their bounding box coordinates 
        bbox_coords_pred, label_pred = model(image_tensor)
        
        plot_prediction()
