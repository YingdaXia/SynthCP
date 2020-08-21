import os.path
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collections import deque

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from options.iounet_options import BaseOptions

from models.fcn8_self_confid import VGG16_FCN8s_SelfConfid
from models.deeplab_self_confid import Deeplab_SelfConfid
import data
import pdb

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
    N x 1 x H x W, where N is batch size.
    Each value is an integer representing correct classification.
    C : integer.
    number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
    N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    target = one_hot.scatter_(1, labels, 1)

    return target

# parse options
opt = BaseOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

###### MODIFY HERE TO USE THE DESIRED MODEL
net = VGG16_FCN8s_SelfConfid(num_cls=opt.label_nc, pretrained=False)
#net = Deeplab_SelfConfid(num_classes=opt.label_nc, init_weights=None, restore_from=None, phase='train')

net.load_state_dict(torch.load(opt.model_path), strict=False)
net.cuda()
transform = []
target_transform = []

#optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.99,
#                        weight_decay=0.0005)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

iteration = 0
losses = deque(maxlen=10)
while True:
    for i, data_i in enumerate(dataloader):
        # Clear out gradients
        optimizer.zero_grad()

        # forward pass and compute loss
        im_src = data_i['image_src'].cuda()
        #im_rec = data_i['image_rec'].cuda()

        iou_label = data_i['iou'].cuda()
        prob = data_i['prob'].cuda()
        label_map = data_i['label_map']
        label_map_no_outlier = torch.tensor(label_map)
        label_map_no_outlier[label_map==19] = 0
        label_map = label_map.cuda()
        labels_hot = make_one_hot (label_map_no_outlier.unsqueeze(1).long(), opt.label_nc).cuda()

        _, conf = net(im_src)

        valid=data_i['valid'].cuda()

        # loss conf
        loss_conf = ((conf - (nn.Softmax(dim=1)(prob) * labels_hot).sum(dim=1)) ** 2) * (label_map.unsqueeze(1) != 19).float()

        loss_conf = loss_conf.mean()
        # backward pass
        loss = loss_conf
        loss.backward()

        # step gradients
        optimizer.step()

        # log results
        if iteration % 10 == 0:
            print('Iteration {}:\tconf loss: {}'.format(iteration, loss_conf.item()))
        iteration += 1

        if iteration % opt.snapshot == 0:
            torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'iter{}.pth'.format(iteration)))
        if iteration >= opt.niter:
            print('Optimization complete.')
            break
    if iteration >= opt.niter:
        break
