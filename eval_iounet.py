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

from models.resnet import IOUNet
import data
import json
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

net = IOUNet(num_cls=opt.label_nc)
net.load_state_dict(torch.load(opt.model_path))
net.eval()
net.cuda()
transform = []
target_transform = []

iteration = 0
losses = deque(maxlen=10)
for i, data_i in enumerate(dataloader):
    # Clear out gradients

    # load data/label
    #im = make_variable(im, requires_grad=False)
    #label = make_variable(label, requires_grad=False)

    # forward pass and compute loss
    im_src = data_i['image_src'].cuda()
    im_rec = data_i['image_rec'].cuda()

    label = data_i['iou'].cuda()
    with torch.no_grad():
        preds = net(im_src, im_rec)
    valid=data_i['valid'].cuda()
    loss = torch.nn.SmoothL1Loss(reduce=False)(preds, label)
    # mask out invalid terms
    loss *= valid.float()
    loss = loss.mean()

    losses.append(loss.item())

    metric = [preds.cpu().numpy()[0].tolist()]
    opt.metric_pred_dir = 'metrics_pred'
    os.makedirs(opt.metric_pred_dir, exist_ok=True)
    with open(os.path.join(opt.metric_pred_dir, os.path.splitext(os.path.basename(data_i['image_src_path'][0]))[0] + '.json'), 'w') as f:
        json.dump(metric, f)

    # log results
    if iteration % 10 == 0:
        print('Iteration {}:\t{}'.format(iteration, np.mean(losses)))
    iteration += 1

