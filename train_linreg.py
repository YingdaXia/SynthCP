import os.path
import os.path as osp
import sys
from collections import deque

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from options.iounet_options import BaseOptions

import data
import pdb

def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss2d(weight=weights, size_average=True,
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score), label)
    return loss

class LinearRegression(nn.Module):

    def __init__(self, num_cls=19):
        super(LinearRegression, self).__init__()
        self.num_cls = num_cls
        self.regress = nn.Linear(self.num_cls, self.num_cls)

    def forward(self, entropy):
        x = self.regress(entropy)

        return x

# parse options
opt = BaseOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

net = nn.Linear(opt.label_nc, opt.label_nc)
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

        # load data/label
        #im = make_variable(im, requires_grad=False)
        #label = make_variable(label, requires_grad=False)

        # forward pass and compute loss

        entropy = data_i['entropy'].cuda()
        iou_label = data_i['iou'].cuda()
        prob = data_i['prob'].cuda()

        pred_iou = net(entropy)

        valid=data_i['valid'].cuda()
        loss_iou = torch.nn.SmoothL1Loss(reduce=False)(pred_iou, iou_label / 100)
        # mask out invalid terms
        loss_iou *= valid.float()
        loss_iou = loss_iou.mean()

        # backward pass
        loss = loss_iou
        loss.backward()

        # step gradients
        optimizer.step()

        # log results
        if iteration % 10 == 0:
            print('Iteration {}:\tiou loss: {}'.format(iteration, loss_iou.item()))
        iteration += 1

        if iteration % opt.snapshot == 0:
            torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'iter{}.pth'.format(iteration)))
        if iteration >= opt.niter:
            print('Optimization complete.')
            break
    if iteration >= opt.niter:
        break
