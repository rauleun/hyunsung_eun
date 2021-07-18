import numpy as np
from layer import *


class MyResNet: 
    def __init__(self, num_classes):
        self.HeadBlock = [
            conv2d(3, 32, 7, 7, stride=2, shift=False),
            bn_layer(32),
            relu(),
            max_pooling(3,3,2,same=True)
        ]
        self.layers = []
        self.layers.append(ResBlock(32, 32, stride=1, shortcut=[conv2d(32, 32, 1, 1, stride=1, shift=False),bn_layer(32)]))
        self.layers.append(ResBlock(32, 32, stride=2, shortcut=[conv2d(32, 32, 1, 1, stride=2, shift=False),bn_layer(32)]))
        self.layers.append(ResBlock(32, 32, stride=1))
        self.layers.append(ResBlock(32, 64, stride=2, shortcut=[conv2d(32, 64, 1, 1, stride=2, shift=False),bn_layer(64)]))
        self.layers.append(ResBlock(64, 64, stride=1))
        self.avg = global_average_pooling()
        self.fc = fully_connected(64, num_classes)

    def train(self):
        self.HeadBlock[1].train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.HeadBlock[1].eval()
        for layer in self.layers:
            layer.eval()

    def forward(self, in_tensor):
        x = in_tensor
        for layer in self.HeadBlock:
            x = layer.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.avg.forward(x)
        out_tensor = self.fc.forward(x)
        
        return out_tensor

    def backward(self, out_diff_tensor, lr):
        x = out_diff_tensor
        self.fc.backward(x, lr)
        x = self.fc.in_diff_tensor
        self.avg.backward(x, lr)
        x = self.avg.in_diff_tensor

        for l in range(1, len(self.layers)+1):
            self.layers[-l].backward(x, lr)
            x = self.layers[-l].in_diff_tensor
        for l in range(1, len(self.HeadBlock)+1):
            self.HeadBlock[-l].backward(x, lr)
            x = self.HeadBlock[-l].in_diff_tensor
        self.in_diff_tensor = x
    
    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], -1)
        return np.argmax(out_tensor, axis=1)

    def save(self, path):
        conv_num = 0
        bn_num = 0
        
        if os.path.exists(path) == False:
            os.mkdir(path)
            
        conv_num = self.HeadBlock[0].save(path, conv_num)
        bn_num = self.HeadBlock[1].save(path, bn_num)

        for layer in self.layers:
            conv_num, bn_num = layer.save(path, conv_num, bn_num)
        self.fc.save(path)

    def load(self, path):
        conv_num = 0
        bn_num = 0

        conv_num = self.HeadBlock[0].load(path, conv_num)
        bn_num = self.HeadBlock[1].load(path, bn_num)

        for layer in self.layers:
            conv_num, bn_num = layer.load(path, conv_num, bn_num)
        self.fc.load(path)
        
        
class ResBlock:

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        self.path1 = [
            conv2d(in_channels, out_channels, 3, 3, stride = stride, shift=False),
            bn_layer(out_channels),
            relu(),
            conv2d(out_channels, out_channels, 3, 3, shift=False),
            bn_layer(out_channels)
        ]
        self.path2 = shortcut
        self.relu = relu()
    
    def train(self):
        self.path1[1].train()
        self.path1[4].train()
        if self.path2 is not None:
            self.path2[1].train()

    def eval(self):
        self.path1[1].eval()
        self.path1[4].eval()
        if self.path2 is not None:
            self.path2[1].eval()

    def forward(self, in_tensor):
        x1 = in_tensor.copy()
        x2 = in_tensor.copy()

        for l in self.path1:
            x1 = l.forward(x1)
        if self.path2 is not None:
            for l in self.path2:
                x2 = l.forward(x2)
        self.out_tensor = self.relu.forward(x1+x2)

        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert self.out_tensor.shape == out_diff_tensor.shape

        self.relu.backward(out_diff_tensor,lr)
        x1 = self.relu.in_diff_tensor
        x2 = x1.copy()

        for l in range(1, len(self.path1)+1):
            self.path1[-l].backward(x1, lr)
            x1 = self.path1[-l].in_diff_tensor

        if self.path2 is not None:
            for l in range(1, len(self.path2)+1):
                self.path2[-l].backward(x2, lr)
                x2 = self.path2[-l].in_diff_tensor
        
        self.in_diff_tensor = x1 + x2

    def save(self, path, conv_num, bn_num):
        conv_num = self.path1[0].save(path, conv_num)
        bn_num = self.path1[1].save(path, bn_num)
        conv_num = self.path1[3].save(path, conv_num)
        bn_num = self.path1[4].save(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].save(path, conv_num)
            bn_num = self.path2[1].save(path, bn_num)

        return conv_num, bn_num

    def load(self, path, conv_num, bn_num):
        conv_num = self.path1[0].load(path, conv_num)
        bn_num = self.path1[1].load(path, bn_num)
        conv_num = self.path1[3].load(path, conv_num)
        bn_num = self.path1[4].load(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].load(path, conv_num)
            bn_num = self.path2[1].load(path, bn_num)

        return conv_num, bn_num
