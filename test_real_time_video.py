# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:43:24 2022

@author: andre

Script to test the model trained in real time with a video
"""

import os
import cv2
import torch
import numpy as np
import configargparse
from vgg16 import VGG16
from torchvision.transforms import transforms


if __name__ == "__main__":
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--log_dir', type=str, default='face_tracking_model', help='Name of the folder to load the model')
    p.add_argument('--path_to_video', type=str, default='videos/video.mp4', help='Path to the video')
    p.add_argument('--checkpoint', type=str, default='checkpoint_43_best.pth',help='Checkpoint')
    p.add_argument('--accuracy', type=float, default=0.94, help='Accuracy threshold')
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
    
    
    cap = cv2.VideoCapture(opt.path_to_video)
    
    if not cap.isOpened():
        raise Exception("Could not open video device")
                    
    cv2.namedWindow("Face Tracking")
    
    # Load the model and create the model
    model = VGG16(in_channels=3, out_channels_bbox=4, out_channels_label=1)
    state_dict = torch.load(os.path.join(opt.log_dir, "checkpoints", opt.checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    transformer = transforms.ToTensor()
    coords = [0, 0, 0, 0]
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        img_resized = cv2.resize(frame, (256, 144))
        image_tensor = transformer(img_resized)
        image_tensor = image_tensor.to(device)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        
        bbox_coords_pred, label_pred = model(image_tensor)
        
        if label_pred.item() > opt.accuracy:
            
            coords[0] = bbox_coords_pred[0][0].item()
            coords[1] = bbox_coords_pred[0][1].item()
            coords[2] = bbox_coords_pred[0][2].item()
            coords[3] = bbox_coords_pred[0][3].item()
            
            cv2.putText(frame, 'Face',
                        tuple(np.multiply(np.array(coords[:2]), [frame.shape[1], frame.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            cv2.rectangle(img=frame,
                          pt1=tuple(np.multiply(np.array(coords[:2]), [frame.shape[1], frame.shape[0]]).astype(int)),
                          pt2=tuple(np.multiply(np.array(coords[2:]), [frame.shape[1], frame.shape[0]]).astype(int)),
                          color=(0, 0, 255),
                          thickness=2)
        
        cv2.imshow("Frame", frame)  
        key = cv2.waitKey(15)

        # Si se presiona la tecla ESC se sale del bucle
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
