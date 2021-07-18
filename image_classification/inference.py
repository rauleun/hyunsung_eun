import cv2
import numpy as np
import csv
import os
import json
import argparse
from tqdm import tqdm

from utils import read_csv, preprocess
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='.', type=str)
parser.add_argument('--csv_pth', default='.', type=str)
parser.add_argument('--category_pth', default='.', type=str)
parser.add_argument('--model_pth', default='.', type=str)
parser.add_argument('--save_pth', default='.', type=str)
parser.add_argument('--batch_size', default=20, type=int)

image_w = 80
image_h = 80

if __name__=="__main__":
    args = parser.parse_args()
    
    with open(args.category_pth) as category_txt:
        category_total = json.load(category_txt)
    datalist = read_csv(args.csv_pth)
    num_classes = len(category_total.keys())
    num_images = len(datalist)
    num_batches = num_images//args.batch_size
    res = num_images-args.batch_size*num_batches

    inputs = []
    targets = []
    labels = np.zeros(num_images, dtype=np.int32)
    infers = np.zeros(num_images, dtype=np.int32)
    for i, data in enumerate(datalist):
        if category_total[data[1]]==1:
            labels[i] = int(category_total[data[1]]-1-1)+num_classes
        else:
            labels[i] = int(category_total[data[1]]-1-1)
        inputs.append(data[0])

    model = MyResNet(num_classes)
    model.load(args.model_pth)
    model.eval()

    for i in tqdm(range(num_batches+1)):
        if i<num_batches:
            batch_inputs = np.zeros((args.batch_size, 3, image_h, image_w))
            for b in range(args.batch_size):
                batch_inputs[b] = preprocess(cv2.imread(os.path.join(args.image_dir, inputs[i*args.batch_size+b])), image_h, image_w)    
            batch_labels = labels[args.batch_size*i:args.batch_size*(i+1)]
            infers[args.batch_size*i:args.batch_size*(i+1)] = model.inference(batch_inputs)
        else:
            if res==0:
                continue
            batch_inputs = np.zeros((res, 3, image_h, image_w))
            for b in range(res):
                batch_inputs[b] = preprocess(cv2.imread(os.path.join(args.image_dir, inputs[i*args.batch_size+b])), image_h, image_w)
            batch_labels = labels[args.batch_size*i:args.batch_size*i+res]
            infers[args.batch_size*i:args.batch_size*i+res] = model.inference(batch_inputs)
        
    category_keys = list(category_total.keys())
    category_keys.append(category_keys[0])
    for i, infer in enumerate(infers):
        targets.append(category_keys[infer+1])

    result_csv = [inputs, targets]
    csvfile = open(args.save_pth, 'w', newline='')
    csvwriter = csv.writer(csvfile)
    for row in zip(*result_csv):
        csvwriter.writerow(row)

    total_acc = np.sum(infers == labels) / infers.shape[0] * 100
    print(f'=> Total accuracy : {total_acc:.3f}')
    class_acc = np.zeros(num_classes)
    cnt = 0
    for c in range(num_classes):
        if np.sum(labels==c)==0:
            class_acc[c] = 0
            cnt += 1
        else:
            class_acc[c] = np.sum((infers==c) & (infers==labels))/np.sum(labels==c) * 100
            print(f'=> Class \'{category_keys[c+1]}\' accuracy :  {np.sum((infers==c) & (infers==labels))/np.sum(labels==c) * 100:.3f}')


