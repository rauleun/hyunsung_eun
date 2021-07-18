import os
import csv
import json
import pickle

from utils import make_category_dict

task1_train_csv, task1_test_csv = '/hd/data/etc/task1_train.csv', '/hd/data/etc/task1_test.csv'
task2_train_csv, task2_test_csv = '/hd/data/etc/task2_train.csv', '/hd/data/etc/task2_test.csv'
task1_save_path = 'task1'
task2_save_path = 'task2'

category_total, category_test = make_category_dict(task1_test_csv, task1_train_csv, task1_save_path)
category_total, category_test = make_category_dict(task2_test_csv, task2_train_csv, task2_save_path)
