import os
import csv
import json
import pickle
from tqdm import tqdm

from utils import log_clip
from dataloader import *
from model import *

def cross_entropy_loss(pred, label):
    batch_size = pred.shape[0]
    loss = np.sum(-label * np.log(pred)-(1-label) * np.log(1 - pred))/batch_size
    diff = (pred - label)/pred/(1 - pred)/batch_size
    return loss, diff

def reverse_cross_entropy_loss(pred, label, A):
    batch_size = pred.shape[0]
    loss = np.sum(-pred * log_clip(label, A)-(1-pred) * log_clip(1-label, A))/batch_size
    diff = (log_clip(1 - label, A)-log_clip(label, A))/batch_size
    return loss, diff


class trainer:
    def __init__(self, model, dataset, num_classes, init_lr):
        self.dataset = dataset
        self.net = model
        self.lr = init_lr
        self.cls_num = num_classes

    def set_lr(self, lr):
        self.lr = lr

    def iterate(self):
        images, labels = self.dataset.get_next_batch()

        out_tensor = self.net.forward(images)
        one_hot_labels = np.eye(self.cls_num)[(labels-1).reshape(-1)].reshape(out_tensor.shape)
            
        loss, out_diff_tensor = cross_entropy_loss(out_tensor, one_hot_labels)
        
        self.net.backward(out_diff_tensor, self.lr)
        
        return loss
    
    def iterate_symmetric(self, A, a, b):
        images, labels = self.dataset.get_next_batch()

        out_tensor = self.net.forward(images)
        one_hot_labels = np.eye(self.cls_num)[(labels-1).reshape(-1)].reshape(out_tensor.shape)
            
        ce_loss, ce_out_diff_tensor = cross_entropy_loss(out_tensor, one_hot_labels)
        rce_loss, rce_out_diff_tensor = reverse_cross_entropy_loss(out_tensor, one_hot_labels, A)
        
        total_loss = a*ce_loss + b*rce_loss
        out_diff_tensor = a*ce_out_diff_tensor + b*rce_out_diff_tensor
        
        self.net.backward(out_diff_tensor, self.lr)
        
        return total_loss
