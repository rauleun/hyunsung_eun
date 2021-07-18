import cv2
import csv
import random
import numpy as np
import os

class dataloader:

    def __init__(self, image_dir, csv_path, category_dict, batch_size, image_w, image_h):
        datalist = []
        new_datalist = []
        with open(csv_path,'r') as csv_data: 
            data_reader = csv.reader(csv_data)
            for data in data_reader:
                datalist.append((data[0], data[1]))          
        datalist = datalist[1:]
        for i, data in enumerate(datalist):
            new_datalist.append((data[0], category_dict[data[1]]-1))
        random.shuffle(new_datalist)
        
        self.image_dir = image_dir
        self.datalist = new_datalist
        self.len = len(self.datalist)
        self.index = 0
        self.batch_size = batch_size
        self.image_w = image_w
        self.image_h = image_h

    def reset(self):
        self.index = 0
        random.shuffle(self.datalist)

    def get_trans_img(self, path):
        img = cv2.imread(os.path.join(self.image_dir, path))
        img = img[:,:,::-1].astype(np.float32).transpose(2,0,1)
        img = img/255
        
        padded_img = np.ones((3, self.image_h, self.image_w))
        if img.shape[2]<self.image_w:
            left_pad = (self.image_w-img.shape[2])//2
            right_pad = self.image_w-img.shape[2]-left_pad
        else:
            left_pad, right_pad = 0, 0
        if img.shape[1]<self.image_h:
            down_pad = (self.image_h-img.shape[1])//2
            up_pad = self.image_h-img.shape[1]-down_pad
        else:
            down_pad, up_pad = 0, 0
        padded_img[:, down_pad:self.image_h-up_pad, left_pad:self.image_w-right_pad] = img
        
        return padded_img

    def get_next_batch(self):
        if self.index + self.batch_size >= self.len:
            self.reset()
        images = np.zeros([self.batch_size, 3, self.image_w, self.image_h],dtype=np.float32)
        labels = np.zeros([self.batch_size],dtype=np.int32)
        for i in range(self.batch_size):
            path, label = self.datalist[i + self.index]
            images[i] = self.get_trans_img(path)
            labels[i] = int(label)
        self.index += self.batch_size
        return images, labels
