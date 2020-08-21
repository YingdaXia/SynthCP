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
import json, pickle
from tqdm import tqdm
from scipy import stats

from options.iounet_options import BaseOptions
from models.resnet import IOUwConfNet, IOUwConfNetBaseline
from util.eval_util import Metrics, eval_ood_measure, eval_alarm_metrics
import data
import pdb

def main():
    # parse options
    opt = BaseOptions().parse()

    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader(opt)

    #net = IOUwConfNetBaseline(num_cls=opt.label_nc)
    #net = IOUwConfNet(num_cls=opt.label_nc)
    net = nn.Linear(opt.label_nc, opt.label_nc)
    net.load_state_dict(torch.load(opt.model_path))
    net.eval()
    net.cuda()
    transform = []
    target_transform = []

    iteration = 0
    losses = deque(maxlen=10)
    aurocs, auprs, fprs = [], [], []
    pred_ious, real_ious = [], []

    metrics = Metrics(
            #['accuracy', 'mean_iou', 'auc', 'ap_success', 'ap_errors', 'fpr_at_95tpr'], 500 * 256 * 512, 19
            ['accuracy', 'mean_iou', 'auc', 'ap_success', 'ap_errors'], 500 * 256 * 512, 19
            )
    for i, data_i in tqdm(enumerate(dataloader)):
        # Clear out gradients

        # load data/label
        #im = make_variable(im, requires_grad=False)
        #label = make_variable(label, requires_grad=False)

        # forward pass and compute loss

        label_map = data_i['label_map'].cuda()
        #pred = prob.argmax(dim=1)

        entropy = data_i['entropy'].cuda()
        iou_label = data_i['iou'].cuda()
        prob = data_i['prob'].cuda()
        max_prob, pred = torch.nn.Softmax(dim=1)(prob).max(dim=1)


        with torch.no_grad():
            pred_iou = net(entropy)


        valid=data_i['valid'][0].cpu().numpy()
        real_iou_valid = iou_label[0].cpu().numpy() / 100
        real_iou_valid[valid==0] = np.nan
        pred_ious.append(pred_iou[0].cpu().numpy())
        real_ious.append(real_iou_valid)

        #metric = [pred_iou.cpu().numpy()[0].tolist()]
        #opt.metric_pred_dir = os.path.join('./checkpoints', opt.name, 'metrics_pred_iouconf')

        #os.makedirs(opt.metric_pred_dir, exist_ok=True)
        #with open(os.path.join(opt.metric_pred_dir, os.path.splitext(os.path.basename(data_i['image_src_path'][0]))[0] + '.json'), 'w') as f:
        #    json.dump(metric, f)
    #scores = metrics.get_scores(split="val")
    #logs_dict = {}
    #for s in scores:
    #    logs_dict[s] = scores[s]
    #print(logs_dict)

    with open(osp.join(osp.dirname(opt.model_path), 'iou_pred_iter{}.pkl'.format(opt.eval_iter)), 'wb') as f:
        pickle.dump({'pred_ious':pred_ious, 'real_ious':real_ious}, f)

    eval_alarm_metrics(pred_ious, real_ious)

if __name__ == '__main__':
    main()
