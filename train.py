# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:46:12 2022

@author: Andres Fandos

Script to load the dataset and train the neural nentwork
"""

import os
import cv2
import json
import utils
from vgg16 import VGG16
import numpy as np
import configargparse
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset


class Dataset(Dataset):
    
    def __init__(self, dataset_path, device):
        
        self.images_path = os.path.join(dataset_path, 'images')
        self.labels_path = os.path.join(dataset_path, 'labels')
        self.images = os.listdir(self.images_path)
        self.transformer = transforms.ToTensor()
        self.device = device
        
        
    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, index):
        
        # Load image
        image = cv2.imread(os.path.join(self.images_path, self.images[index]))
        image = self.transformer(image)
        image = image.to(self.device)
        
        with open(os.path.join(self.labels_path,
                               self.images[index].split('.')[0] + '.' +
                               self.images[index].split('.')[1] + '.json'), 'r') as f:
            labels = json.load(f)
        
        # Load coordinates of the bounding box
        coords = torch.zeros(4)
        coords[0] = labels["bbox"][0]
        coords[1] = labels["bbox"][1]
        coords[2] = labels["bbox"][2]
        coords[3] = labels["bbox"][3]
        coords = coords.to(self.device)
        
        # Load label
        label = torch.zeros(1)
        label[0] = labels["class"]
        label = label.to(self.device)
        
        return image, coords, label
    


def train(num_epochs, batch_size, learning_rate, log_dir):
    
    min_val_loss = np.inf
    bestEpoch = 0

    train_accuracies = []
    val_accuracies = []    
    train_losses = []
    val_losses = []
    
    
    print('--------------------------------------------------------------')
    
    # Loop along epochs to do the training
    for i in range(num_epochs + 1):
        
        print(f'EPOCH {i}')
        
        # Training loop
        train_acc = 0.0
        train_loss = 0.0
        model.train()
        iteration = 1
        
        print('\nTRAINING')
        
        for images, bbox_coords, labels in train_loader:
            
            print('\rEpoch[' + str(i) + '/' + str(num_epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(train_loader)), end='')
            iteration += 1
            
            images, bbox_coords, labels = images.to(device), bbox_coords.to(device), labels.to(device)
            
            optimiser.zero_grad()
            
            bbox_coords_pred, labels_pred = model(images)
            
            loss_classification = loss_fn_classification(labels_pred, labels)
            loss_bbox_coords = loss_fn_bbox_coords(bbox_coords_pred, bbox_coords)
            total_loss = loss_bbox_coords + 0.25*loss_classification
            
            total_loss.backward()
            optimiser.step()
            
            train_acc += accuracy(labels_pred, labels).item()
            train_loss += total_loss.item()
        
        
        # Validation loop
        val_acc = 0.0
        val_loss = 0.0
        model.eval()
        iteration = 1

        print('')
        print('\nVALIDATION')
        
        for images, bbox_coords, labels in validate_loader:
            
            print('\rEpoch[' + str(i) + '/' + str(num_epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(validate_loader)), end='')
            iteration += 1
            
            images, bbox_coords, labels = images.to(device), bbox_coords.to(device), labels.to(device)
            
            bbox_coords_pred, labels_pred = model(images)
            
            loss_classification = loss_fn_classification(labels_pred, labels)
            loss_bbox_coords = loss_fn_bbox_coords(bbox_coords_pred, bbox_coords)
            total_loss = loss_bbox_coords + 0.25*loss_classification
            
            val_acc += accuracy(labels_pred, labels).item()
            val_loss += total_loss.item()
        

        # Save loss and accuracy values
        train_accuracies.append(train_acc / len(train_loader))
        val_accuracies.append(val_acc / len(validate_loader))
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(validate_loader))
        
        print('\n')
        print(f'- Train accuracy: {(train_acc / len(train_loader))*100:.3f}%')
        print(f'- Validation accuracy: {(val_acc / len(validate_loader))*100:.3f}%')
        print(f'- Train loss: {train_loss / len(train_loader):.3f}')
        print(f'- Validation loss: {val_loss / len(validate_loader):.3f}')
        
            
        # Save the model every 10 epochs
        if i % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + ".pth"))
            
        # Save the best model when loss decreases
        if (val_loss / len(validate_loader)) < min_val_loss:
            
            # If first epoch, save model as best, otherwise, replace the previous best model with the current one
            if i == 0:
                torch.save(model.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best.pth"))
            else:
                os.remove(os.path.join(checkpoints_path, "checkpoint_" + str(bestEpoch) + "_best.pth"))
                torch.save(model.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best.pth"))
            
            print(f'\nValidation loss decreased: {min_val_loss:.3f} ---> {val_loss / len(validate_loader):.3f}\nModel saved')
                
            # Update parameters with the new best model
            min_val_loss = val_loss / len(validate_loader)
            bestEpoch = i
            
        save_loss_acc(log_dir, np.array(train_accuracies), np.array(val_accuracies),
                      np.array(train_losses), np.array(val_losses))
            
        print("--------------------------------------------------------------")
    
    # Plot loss and accuracy curves
    utils.plot_loss(log_dir, np.array(train_losses), np.array(val_losses),
                    np.array(train_accuracies), np.array(val_accuracies))



if __name__ == "__main__":
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str, default='aug_data', help='Dataset path.')
    p.add_argument('--train_split', type=float, default=0.85, help='Percentage of the dataset to use for training.')
    p.add_argument('--log_dir', type=str, default='face_tracking_model', help='Name of the folder to save the model.')
    p.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    p.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    p.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    p.add_argument('--device', type=str, default='gpu', help='Choose the device to train the model: "gpu" or "cpu"')
    opt = p.parse_args()
    
    assert not (os.path.isdir(opt.log_dir)), 'The folder log_dir already exists, remove it or change it'
    assert (opt.train_split < 1), 'The percentage of the dataset to use for training must be lower than 1'
    
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
            
         
    # Load datasets and create dataloaders
    dataset = Dataset(opt.dataset_path, device)
    
    # # Split the dataset in train, validation and test
    num_images = len(dataset)
    models_training = int(num_images * opt.train_split)
    models_validation = num_images - models_training
    
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [models_training,  models_validation])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=opt.batch_size, shuffle=True)
    
    print('Images used to train: ' + str(len(train_dataset)) + '/' + str(len(dataset)))
    print('Images used to validate: ' + str(len(validate_dataset)) + '/' + str(len(dataset)) + '\n')
    
    model = VGG16(in_channels=train_dataset[0][0].size(dim=0),
                  out_channels_bbox=train_dataset[0][1].size(dim=0),
                  out_channels_label=train_dataset[0][2].size(dim=0)).to(device)
    
    checkpoints_path = utils.create_model_folder(opt.log_dir)
    optimiser = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-4)
    loss_fn_classification = nn.BCELoss()
    loss_fn_bbox_coords = utils.loss_fn_bbox_coords
    accuracy = utils.accuracy
    save_loss_acc = utils.save_loss_acc
    
    train(opt.epochs, opt.batch_size, opt.learning_rate, opt.log_dir)
