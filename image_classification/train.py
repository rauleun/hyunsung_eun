import os
import csv
import json
import pickle
import argparse
from tqdm import tqdm

from utils import save_value
from dataloader import *
from model import *
from evaluate import *
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='.', type=str)
parser.add_argument('--train_csv_pth', default='.', type=str)
parser.add_argument('--test_csv_pth', default='.', type=str)
parser.add_argument('--category_pth', default='.', type=str)
parser.add_argument('--model_save_pth', default='.', type=str)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--symmetric_training', default=False, type=bool)
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    image_h = 80
    image_w = 80
    if os.path.exists(args.model_save_pth) == False:
        os.mkdir(args.model_save_pth)
    with open(args.category_pth) as category_txt:
        category_total = json.load(category_txt)
    dataset = dataloader(args.image_dir, args.train_csv_pth, category_total, args.batch_size, image_w, image_h)
    num_classes = len(category_total.keys())
    model = MyResNet(num_classes)

    init_lr = 0.05
    train = trainer(model, dataset, num_classes, init_lr)
    
    loss_value, acc_value = [], []
    loss_path, acc_path = 'loss.txt', 'accuracy.txt'
    temp_loss = 0
    best_acc = 0
    interval_loss = 2
    interval_acc = 100
    total_steps = args.epochs*dataset.len//args.batch_size
    print(f"Total steps : {total_steps}")

    for step in tqdm(range(total_steps)):
        if args.symmetric_training:
            temp_loss += train.iterate()
        else:
            A, a, b = -6, 0.1, 1
            temp_loss += train.iterate_symmetric(A, a, b)
        if step % interval_loss == 0 and step != 0:
            loss_value.append((step, temp_loss / interval_loss))  
            save_value(os.path.join(args.model_save_pth, loss_path), loss_value)
            print("iteration = {} || loss = {}".format(str(step), str(temp_loss/interval_loss)))
            temp_loss = 0
            if step % interval_acc == 0:              
                model.eval()
                total_acc, class_acc = evaluate(model, args.image_dir, args.test_csv_pth, category_total ,args.batch_size, image_w, image_h)
                print(f"iteration = {step} || accuracy = {total_acc}")
                acc_value.append([step, total_acc] + class_acc.tolist())
                save_value(os.path.join(args.model_save_pth, acc_path), acc_value)
                if total_acc>best_acc:
                    best_acc = total_acc
                    model.save(os.path.join(args.model_save_pth, f"model{step}"))                  
                model.train()
        if step == 500:
            train.set_lr(0.02)        
        if step == 1000:
            train.set_lr(0.01)
        if step == 2000:
            train.set_lr(0.005)
        if step == 5000:
            train.set_lr(0.001)
