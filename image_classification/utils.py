import csv
import pickle
import numpy as np


def save_value(path, value):
    with open(path, 'wb') as file:
        pickle.dump(value, file)
        
def log_clip(tensor, A):
    tensor[tensor==0] = np.exp(A)
    return np.log(tensor)

def preprocess(img, image_h, image_w):
    img = img[:,:,::-1].astype(np.float32).transpose(2,0,1)       
    img = img/255
    return padding(image_h, image_w, img)

def padding(image_h, image_w, img):
    padded_img = np.ones((3, image_h, image_w))
    if img.shape[2]<image_w:
        left_pad = (image_w-img.shape[2])//2
        right_pad = image_w-img.shape[2]-left_pad
    else:
        left_pad, right_pad = 0, 0
    if img.shape[1]<image_h:
        down_pad = (image_h-img.shape[1])//2
        up_pad = image_h-img.shape[1]-down_pad
    else:
        down_pad, up_pad = 0, 0
    padded_img[:, down_pad:image_h-up_pad, left_pad:image_w-right_pad] = img
    return padded_img

def read_csv(csv_path):
    data_list = []
    with open(csv_path,'r') as csv_data: 
        data_reader = csv.reader(csv_data)
        for data in data_reader:
            data_list.append((data[0], data[1]))     
        data_list = data_list[1:]
    return data_list
        
def make_category_dict(csv_test, csv_train, save_path):   
    category_test = list(set(read_csv(csv_test)))
    category_train = list(set(read_csv(csv_train)))
    category_total = category_test.copy()
    for cat in category_train:
        if cat not in category_test:
            category_total.append(cat)         
    category_dict_total, category_dict_test = {}, {}
    for i, cat in enumerate(category_total):
        category_dict_total[cat] = i+1
    for i, cat in enumerate(category_test):
        category_dict_test[cat] = i+1
    with open(os.path.join(save_path, 'category.txt'), 'w') as file:
        file.write(json.dumps(category_dict_total))
    with open(os.path.join(save_path, 'category_val.txt'), 'w') as file:
        file.write(json.dumps(category_dict_test))
    return category_dict_total, category_dict_test