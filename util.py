#coding=utf-8
# util.py
# python Version: python3.6
# by Zhongang Qi (qiz@oregonstate.edu)
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage import filters
from collections import OrderedDict

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor



def topmaxPixel(HattMap, thre_num):
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    #print(ii)
    OutHattMap = HattMap*0
    OutHattMap[ii] = 1

    img_ratio = np.sum(OutHattMap) / OutHattMap.size
    #print(OutHattMap.size)
    OutHattMap = 1 - OutHattMap


    return OutHattMap, img_ratio


def topmaxPixel_insertion(HattMap, thre_num):
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    # print(ii)
    OutHattMap = HattMap * 0
    OutHattMap[ii] = 1

    img_ratio = np.sum(OutHattMap) / OutHattMap.size

    return OutHattMap, img_ratio




def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad




def preprocess_image(img, use_cuda=1, require_grad = False):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=require_grad)



def numpy_to_torch(img, use_cuda=1, requires_grad=False):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v



def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        module.bias.data.fill_(0.01)

class qFunction(nn.Module):
    """ Model for q-function """
    
    def __init__(self):
        super(qFunction, self).__init__()
        
        self.input_len = 68
        self.output_len = 8
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_len, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_len)
        )
    
        self.softmax_func = nn.Softmax()

        self.layers.apply(weights_initialize)
        
    def forward(self, input):
        x = input
        for layer in self.layers:
            if type(layer) == nn.Linear:
                x = x.view(-1, int(np.prod(x.shape[1:])))
            x = layer(x)
            
        x = self.softmax_func(x)
        return x

def predict(model, input, steps):
    q_values = model(input).squeeze(1)

    return q_values

def predict_batch(model, input):
    if use_cuda:
        input = input.cuda()
    q_values = model(input)
#         values, q_actions = q_values.max(1)
    return q_values




def load_model_new(use_cuda = 1, model_name = 'resnet50'):

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_name == 'q_function':
        qmodel = qFunction()

        qtest = torch.load("q_model_decom8_grid.pt", map_location = 'cpu')

        new_dict = OrderedDict()
        new_keys = list(qmodel.state_dict().keys())

        for i, (key, weight) in enumerate(qtest.items()):
            new_dict[new_keys[i]] = weight

        model = qmodel.load_state_dict(new_dict)


    #print(model)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model


def save_heatmap(output_path, mask, img, blurred, blur_mask=0):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask))
    mask = 1 - mask

    if blur_mask:
        mask = cv2.GaussianBlur(mask, (11, 11), 10)
        mask = np.expand_dims(mask, axis=2)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255


    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    IGOS = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8)* heatmap;



    cv2.imwrite(output_path + "heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(output_path + "IGOS.png", np.uint8(255 * IGOS))
    cv2.imwrite(output_path + "blurred.png", np.uint8(255 * blurred))
