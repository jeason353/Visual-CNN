import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
import cv2

import sys
import os
import pdb

from models import *

np.random.seed(0)
torch.manual_seed(1)

def load_image(path):
    img = cv2.imread(path)
    # print(img.shape)
    img = cv2.resize(img, (224,224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(img)
    img.unsqueeze_(0)
    return img


if __name__ == '__main__':
    img = load_image('data/cat.jpg')
    model = net()

    model.load_state_dict(torch.load('vgg16-397923af.pth'))
    de_model = denet(model)
    out = model(img)
    

    feature_map = model.feature_maps[10].clone()

    max_lst = []
    for i in range(feature_map.size()[1]):
        max_lst.append(feature_map[0,i,:,:].max())
    max_loc = np.argmax(np.array(max_lst))
    if max_loc == 0:
        feature_map[0, 1:, :, :] = 0
    elif max_loc == feature_map.size()[1] - 1:
        feature_map[0, :-1, :, :] = 0
    else:
        feature_map[0, :max_loc, :, :] = 0
        feature_map[0, max_loc:, :, :] = 0

    with torch.no_grad():
        img_1 = de_model(feature_map, 20, model.pool_indices)
    img_1 = img_1.numpy()[0].transpose(1,2,0)
    img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min()) * 255
    img_1 = (img_1+img_1.min()).astype(np.uint8)
    cv2.imshow('image', img_1)
    cv2.waitKey()
