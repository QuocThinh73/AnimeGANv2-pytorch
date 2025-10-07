from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys

VGG_MEAN = [103.939, 116.779, 123.68] # B, G, R

class VGG16(nn.Module):
    def __init__(self, vgg16_npy_path='vgg16_weight/vgg16.npy', include_fc=False):
        super().__init__()
        if vgg16_npy_path:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
            print(f"Successfully to load {vgg16_npy_path} file")
        else:
            self.data_dict = None
            print(f"Failed to load {vgg16_npy_path} file")
            sys.exit(1)
        self.include_fc = include_fc
        self.register_buffer("vgg_mean", torch.tensor(VGG_MEAN, dtype=torch.float32).view(1, 3, 1, 1))

    def _get_convolutional_parameters(self, name, x):
        w_np = self.data_dict[name][0]
        b_np = self.data_dict[name][1]
        w = torch.from_numpy(w_np).permute(3, 2, 0, 1).contiguous().to(dtype=x.dtype, device=x.device)
        b = torch.from_numpy(b_np).to(dtype=x.dtype, device=x.device)
        return w, b
    
    def _convolution_no_activation(self, x, name):
        w, b = self._get_convolutional_parameters(name, x)
        padding = w.shape[-1] // 2
        return F.conv2d(x, w, b, stride=1, padding=padding)
    
    def _convolution_layer(self, x, name):
        return F.relu(self._convolution_no_activation(x, name), inplace=False)
    
    def _max_pool(self, x, name):
        return F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
    
    def _fc_layer(self, x, name):
        w_np = self.data_dict[name][0]
        b_np = self.data_dict[name][1]
        w = torch.from_numpy(w_np).t().contiguous().to(dtype=x.dtype, device=x.device)
        b = torch.from_numpy(b_np).to(dtype=x.dtype, device=x.device)
        return F.linear(x, w, b)
    
    def forward(self, rgb, return_dict=True):
        start = time.start()
        if rgb.dim() == 4 and rgb.shape[1] != 3 and rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2).contiguous()
        
        x = ((rgb + 1.0) / 2.0) * 255.0 # [-1, 1] -> [0, 255]
        x = x[:, [2, 1, 0], :, :] # RGB -> BGR
        x = x - self.vgg_mean.to(x.dtype) # substract mean
        features = {}

        # VGG16 backbone (13 conv)
        conv1_1 = self._convolution_layer(x, "conv1_1")
        features["conv1_1"] = conv1_1
        conv1_2 = self._convolution_layer(conv1_1, "conv1_2")
        features["conv1_2"] = conv1_2
        pool1 = self._max_pool(conv1_2, "pool1")
        features["pool1"] = pool1

        conv2_1 = self._convolution_layer(pool1, "conv2_1")
        features["conv2_1"] = conv2_1
        conv2_2 = self._convolution_layer(conv2_1, "conv2_2")
        features["conv2_2"] = conv2_2
        pool2 = self._max_pool(conv2_2, "pool2")
        features["pool2"] = pool2

        conv3_1 = self._convolution_layer(pool2, "conv3_1")
        features["conv3_1"] = conv3_1
        conv3_2 = self._convolution_layer(conv3_1, "conv3_2")
        features["conv3_2"] = conv3_2
        conv3_3 = self._convolution_layer(conv3_2, "conv3_3")
        features["conv3_3"] = conv3_3
        pool3 = self._max_pool(conv3_3, "pool3")
        features["pool3"] = pool3

        conv4_1 = self._convolution_layer(pool3, "conv4_1")
        features["conv4_1"] = conv4_1
        conv4_2 = self._convolution_layer(conv4_1, "conv4_2")
        features["conv4_2"] = conv4_2

        conv4_3_no_activation = self._convolution_no_activation(conv4_2, "conv4_3")
        features["conv4_3_no_activation"] = conv4_3_no_activation

        conv4_3 = F.relu(conv4_3_no_activation, inplace=False)
        features["conv4_3"] = conv4_3
        pool4 = self._max_pool(conv4_3, "pool4")
        features["pool4"] = pool4

        conv5_1 = self._convolution_layer(pool4, "conv5_1")
        features["conv5_1"] = conv5_1
        conv5_2 = self._convolution_layer(conv5_1, "conv5_2")
        features["conv5_2"] = conv5_2
        conv5_3 = self._convolution_layer(conv5_2, "conv5_3")
        features["conv5_3"] = conv5_3
        pool5 = self._max_pool(conv5_3, "pool5")
        features["pool5"] = pool5

        out = pool5
        if self.include_fc:
            n = out.shape[0]
            out = out.view(n, -1)
            fc6 = self._fc_layer(pool5, "fc6")
            features["fc6"] = fc6
            relu6 = F.relu(fc6, inplace=False)
            features["relu6"] = relu6
            fc7 = self._fc_layer(relu6, "fc7")
            features["fc7"] = fc7
            relu7 = F.relu(fc7, inplace=False)
            features["relu7"] = relu7
            fc8= self._fc_layer(relu7, "fc8")
            features["fc8"] = fc8
            probs = F.softmax(fc8, dim=1)
            features["prob"] = probs
            out = probs
        
        return features if return_dict else out