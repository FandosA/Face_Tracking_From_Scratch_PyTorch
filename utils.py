# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:30:32 2022

@author: Andres Fandos

Useful functions to use in the train and test scripts
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
    

def create_model_folder(log_dir):
    
    os.mkdir(log_dir)
    
    checkpoints_path = os.path.join(log_dir, 'checkpoints')
    os.mkdir(checkpoints_path)

    return checkpoints_path


def accuracy(pred, groud_truth):
    
    pred = pred >= 0.5
    truth = groud_truth >= 0.5
    acc = pred.eq(truth).sum() / groud_truth.numel()
    
    return acc


def loss_fn_bbox_coords(predicts, y_true):
        
    box_center_error = torch.sum(torch.square(y_true[:, :2] - predicts[:, :2]))
                  
    h_true = y_true[:, 3] - y_true[:, 1] 
    w_true = y_true[:, 2] - y_true[:, 0] 

    h_pred = predicts[:, 3] - predicts[:, 1] 
    w_pred = predicts[:, 2] - predicts[:, 0] 
    
    box_size_error = torch.sum(torch.square(w_true - w_pred) + torch.square(h_true - h_pred))
    
    return box_center_error + box_size_error

    
def save_loss_acc(log_dir, train_accuracies, val_accuracies, train_losses, val_losses):
    
    np.savetxt(os.path.join(log_dir, 'train_accuracies.txt'), train_accuracies)
    np.savetxt(os.path.join(log_dir, 'val_accuracies.txt'), val_accuracies)
    np.savetxt(os.path.join(log_dir, 'train_losses.txt'), train_losses)
    np.savetxt(os.path.join(log_dir, 'val_losses.txt'), val_losses)

    
def plot_loss(log_dir, train_losses, validation_losses, train_accs, validation_accs):
    
    epochs = np.arange(train_losses.shape[0])
    bestEpoch = np.argmin(validation_losses)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss", c='b')
    plt.plot(epochs, validation_losses, label="Validation loss", c='r')
    plt.plot(bestEpoch, validation_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.01, validation_losses[bestEpoch]+.01, str(bestEpoch) + ' - ' + str(round(validation_losses[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, 'loss.png'))    
    
    plt.figure()
    plt.plot(epochs, train_accs, label="Training accuracy", c='b')
    plt.plot(epochs, validation_accs, label="Validation accuracy", c='r')
    plt.plot(bestEpoch, validation_accs[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.001, validation_accs[bestEpoch]+.001, str(bestEpoch) + ' - ' + str(round(validation_accs[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, 'accuracy.png'))
    
    plt.show()
