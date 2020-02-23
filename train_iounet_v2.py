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

from models.resnet import IOUwConfNet, IOUwConfNetBaseline
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

# parse options
opt = BaseOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

net = IOUwConfNet (num_cls=opt.label_nc)
#net = IOUwConfNetBaseline (num_cls=opt.label_nc)
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
        im_src = data_i['image_src'].cuda()
        im_rec = data_i['image_rec'].cuda()

        iou_label = data_i['iou'].cuda()
        prob = data_i['prob'].cuda()
        label_map = data_i['label_map'].cuda()

        h, w = im_rec.shape[-2:]
        #prob = torch.nn.functional.interpolate(prob, (h, w), mode='bilinear')
        #label_map = torch.nn.functional.interpolate(label_map.unsqueeze(0).float(), (h, w)).byte()

        correct_map = (prob.argmax(dim=1).long() == label_map.long()).float()

        pred_iou, conf = net(prob, im_src, im_rec)

        valid=data_i['valid'].cuda()
        loss_iou = torch.nn.SmoothL1Loss(reduce=False)(pred_iou, iou_label / 100)
        # mask out invalid terms
        loss_iou *= valid.float()
        loss_iou = loss_iou.mean()

        # loss conf
        loss_conf = torch.nn.BCELoss(reduction='none')(conf, correct_map.unsqueeze(1)) * (label_map.unsqueeze(1) != 19).float()
        loss_conf = loss_conf * (torch.nn.Softmax(dim=1)(prob).max(dim=1, keepdim=True)[0] ** 2)
        loss_conf = loss_conf.mean()
        # backward pass
        loss = loss_iou + loss_conf
        loss.backward()

        # step gradients
        optimizer.step()

        # log results
        if iteration % 10 == 0:
            #print('Iteration {}:\tiou loss: {} \tconf loss: {}'.format(iteration, loss_iou.item(), loss_conf.item()))
            print('Iteration {}:\tiou loss: {} '.format(iteration, loss_iou.item()))
        iteration += 1

        if iteration % opt.snapshot == 0:
            torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'iter{}.pth'.format(iteration)))
        if iteration >= opt.niter:
            print('Optimization complete.')
            break
    if iteration >= opt.niter:
        break
