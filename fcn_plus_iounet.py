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
from options.fcn_iounet_options import BaseOptions
from models.networks.generator import SPADEGenerator
from models.fcn8 import VGG16_FCN8s
from models.resnet import IOUNet
import os
import json
import pdb

import data

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

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def result_stats(hist):
    acc_overall = np.diag(hist).sum() / hist.sum() * 100
    acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) * 100
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    freq = hist.sum(1) / hist.sum()
    fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
    pix_percls = hist.sum(1)
    return acc_overall, acc_percls, iu, fwIU, pix_percls

# parse options
opt = BaseOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

fcn = VGG16_FCN8s(num_cls=opt.label_nc, pretrained=False)
fcn.load_state_dict(torch.load(opt.fcn_model_path))
#fcn.train()
fcn.eval()
fcn.cuda()

netG = SPADEGenerator(opt)
netG.load_state_dict(torch.load(opt.SPADE_model_path))
netG.eval()
netG.cuda()

iounet = IOUNet(num_cls=opt.label_nc)
iounet.load_state_dict(torch.load(opt.IOUNet_model_path))
#iounet.eval()
iounet.cuda()

resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

optimizer = torch.optim.SGD(fcn.parameters(), lr=0.0000001, momentum=0.99,
                        weight_decay=0.0005)

iteration = 0
losses = deque(maxlen=10)
while True:
    for i, data_i in enumerate(dataloader):
        # Clear out gradients
        #optimizer.zero_grad()

        # load data/label
        #im = make_variable(im, requires_grad=False)
        #label = make_variable(label, requires_grad=False)

        while True:
            #optimizer.zero_grad()

            # forward pass and compute loss
            im = data_i['image_seg'].cuda()
            im_origin = data_i['image'].cuda()
            label = data_i['label'].cuda().long().squeeze(1)

            # get label mask to mask out non-existing classes
            mask = torch.zeros((1,19)).cuda()
            for i in range(19):
                mask[0,i] = ((label == i).sum() > 0).float()

            # test fcn
            pred_fcn = fcn(im)
            pred_fcn = torch.nn.Softmax(dim=1)(pred_fcn)

            _, preds = torch.max(pred_fcn, 1)

            hist = fast_hist(label.cpu().numpy().flatten(),
                preds.cpu().numpy().flatten(),
                19)
            acc_overall, acc_percls, iu, fwIU, pix_percls = result_stats(hist)
            #print(iu[1])
            #pdb.set_trace()
            #Image.fromarray(np.uint8(preds.cpu().numpy()[0])).save(os.path.splitext(os.path.basename(data_i['path'][0]))[0] + '_pred.png')
            # test spade
            one_hot_input = torch.zeros_like(pred_fcn).cuda()
            for i in range(19):
                one_hot_input[:,i,:,:] = (preds == i).float()

            #pred_spade = netG(pred_fcn)
            pred_spade = netG(one_hot_input)

            # adjust normalization for rec and source
            pred_spade = ((pred_spade + 1) / 2.0).clamp(0, 1)
            #Image.fromarray(np.transpose(pred_spade[0].detach().cpu().numpy().astype(np.uint8), [1,2,0])).save(os.path.splitext(os.path.basename(data_i['path'][0]))[0] + '_rec.png')
            
            im_origin = ((im_origin + 1) / 2.0).clamp(0, 1)
            pred_spade = resnet_transform(pred_spade.squeeze(0)).unsqueeze(0)
            im_origin = resnet_transform(im_origin.squeeze(0)).unsqueeze(0)


            # test iou
            iounet.eval()
            pred_iou = iounet(im_origin, pred_spade)
            print(pred_iou)
            input()
            break
            

            #metric = [pred_iou.detach().cpu().numpy()[0].tolist()]
            #opt.metric_pred_dir = 'metrics_pred'
            #os.makedirs(opt.metric_pred_dir, exist_ok=True)
            #with open(os.path.join(opt.metric_pred_dir, os.path.splitext(os.path.basename(data_i['path'][0]))[0] + '.json'), 'w') as f:
            #    json.dump(metric, f)

        # log results
        #if iteration % 10 == 0:
        #    print('Iteration {}:\t{}'.format(iteration, np.mean(losses)))
        #iteration += 1


        #loss = torch.nn.CrossEntropyLoss(ignore_index=opt.label_nc)(preds, label)
        
        # backward pass
        #loss.backward()
        #losses.append(loss.item())

        # step gradients
        #optimizer.step()

        # log results
        #if iteration % 10 == 0:
        #    print('Iteration {}:\t{}'.format(iteration, np.mean(losses)))
        #iteration += 1

        #if iteration % opt.snapshot == 0:
        #    torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, '{}-iter{}.pth'.format(opt.name, iteration)))
        #if iteration >= opt.niter:
        #    print('Optimization complete.')
        #    break
    #if iteration >= opt.niter:
    #    break
    break
