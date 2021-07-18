import os
import cv2
import csv
import numpy as np

from utils import read_csv, preprocess

def evaluate(model, image_dir, test_csv_pth, category_dict, batch_size, image_w, image_h):
    model.eval()
    datalist = []
    num_classes = len(category_dict.keys())
    datalist = read_csv(test_csv_pth)
    num_images = len(datalist)
    num_batches = num_images//batch_size
    res = num_images-batch_size*num_batches
    
    inputs = []
    targets = []
    labels = np.zeros(num_images, dtype=np.int32)
    infers = np.zeros(num_images, dtype=np.int32)
    for i, data in enumerate(datalist):
        if category_dict[data[1]]==1:
            labels[i] = int(category_dict[data[1]]-1-1)+num_classes
        else:
            labels[i] = int(category_dict[data[1]]-1-1)
        inputs.append(data[0])

    for i in range(num_batches+1):
        if i<num_batches:
            batch_inputs = np.zeros((batch_size, 3, image_h, image_w))
            for b in range(batch_size):
                batch_inputs[b] = preprocess(cv2.imread(os.path.join(image_dir, inputs[i*batch_size+b])), image_h, image_w)
            batch_labels = labels[batch_size*i:batch_size*(i+1)]
            infers[batch_size*i:batch_size*(i+1)] = model.inference(batch_inputs)
        else:
            if res==0:
                continue
            batch_inputs = np.zeros((res, 3, image_h, image_w))
            for b in range(res):
                batch_inputs[b] = preprocess(cv2.imread(os.path.join(image_dir, inputs[i*batch_size+b])), image_h, image_w)
            batch_labels = labels[batch_size*i:batch_size*i+res]
            infers[batch_size*i:batch_size*i+res] = model.inference(batch_inputs)
            
    total_acc = np.sum(infers == labels) / infers.shape[0] * 100
    class_acc = np.zeros(num_classes)
    cnt = 0
    for c in range(num_classes):
        if np.sum(labels==c)==0:
            class_acc[c] = 0
            cnt += 1
        else:
            class_acc[c] = np.sum((infers==c) & (infers==labels))/np.sum(labels==c) * 100
        
    return total_acc, class_acc[:num_classes-cnt]
