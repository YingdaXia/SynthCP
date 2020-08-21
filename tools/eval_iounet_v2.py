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

import data
from options.iounet_options import BaseOptions
from models.resnet import IOUwConfNet, IOUwConfNetBaseline

from util.eval_util import Metrics, eval_ood_measure, eval_alarm_metrics
import pdb

def main():
    # parse options
    opt = BaseOptions().parse()
    out_path = osp.join(osp.dirname(opt.iou_dir), 'conf_aupr_maxprob')
    os.makedirs(out_path, exist_ok=True)

    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader(opt)

    # Use the following line to use Direct Prediction
    #net = IOUwConfNetBaseline(num_cls=opt.label_nc)
    # Use the following line to use SynthCP
    net = IOUwConfNet(num_cls=opt.label_nc)

    net.load_state_dict(torch.load(opt.model_path))
    net.eval()
    net.cuda()
    transform = []
    target_transform = []

    aurocs, auprs, fprs = [], [], []
    pred_ious, real_ious = [], []

    metrics = Metrics(
            ['accuracy', 'mean_iou', 'auc', 'ap_success', 'ap_errors'], 500 * 256 * 512, 19
            )
    for i, data_i in tqdm(enumerate(dataloader)):

        im_src = data_i['image_src'].cuda()
        im_rec = data_i['image_rec'].cuda()

        iou_label = data_i['iou'].cuda()
        prob = data_i['prob'].cuda()
        label_map = data_i['label_map'].cuda()
        max_prob, pred = torch.nn.Softmax(dim=1)(prob).max(dim=1)

        with torch.no_grad():
            pred_iou, conf = net(prob, im_src, im_rec)

        max_prob_to_save = max_prob.cpu().numpy()
        conf_to_save = conf.squeeze(0).cpu().numpy()

        pred = pred[label_map != 19]
        conf = conf.squeeze(0)
        conf = conf[label_map != 19]
        max_prob = max_prob[label_map != 19]
        label_map = label_map[label_map != 19]

        correct_map = (pred.long() == label_map.long()).float()

        # !!!!!!!! Use the following line for SynthCP        !!!!!!!!!
        conf = conf + max_prob
        # !!!!!!!! Use the following line for MSP Evaluation !!!!!!!!!
        #conf = max_prob

        res = eval_ood_measure(conf.cpu().numpy(), pred.cpu().numpy(), label_map.cpu().numpy(), mask=None)
        if res is not None:
            auroc, aupr, fpr = res
            aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)

        metrics.update(pred.long(), label_map.long(), conf)

        valid=data_i['valid'][0].cpu().numpy()
        real_iou_valid = iou_label[0].cpu().numpy() / 100
        real_iou_valid[valid==0] = np.nan
        pred_ious.append(pred_iou[0].cpu().numpy())
        real_ious.append(real_iou_valid)

        #conf_dir = os.path.join('./checkpoints', opt.name, 'confnetpred')
        #os.makedirs(conf_dir, exist_ok=True)
        #np.savez_compressed(os.path.join(conf_dir, os.path.basename(data_i['image_src_path'][0])),
        #                    conf=conf[0].cpu().numpy(), prob=prob[0].cpu().numpy(), label=label_map[0].cpu().numpy())

        #metric = [pred_iou.cpu().numpy()[0].tolist()]
        #opt.metric_pred_dir = os.path.join('./checkpoints', opt.name, 'metrics_pred_iouconf')
        #os.makedirs(opt.metric_pred_dir, exist_ok=True)
        #with open(os.path.join(opt.metric_pred_dir, os.path.splitext(os.path.basename(data_i['image_src_path'][0]))[0] + '.json'), 'w') as f:
        #    json.dump(metric, f)

        # Things to save for visualization: conf_to_save, max_prob_to_save, aupr, aupr_msp
        #case_name = osp.splitext(osp.basename(data_i['image_src_path'][0]))[0]
        #with open(osp.join(out_path, case_name+'.pkl'), 'wb') as f:
        #    pickle.dump(dict(conf=conf_to_save, max_prob=max_prob_to_save, aupr=aupr, aupr_msp=aupr_msp), f)

    
    print(" mean fpr = ", np.mean(fprs))
    scores = metrics.get_scores(split="val")
    logs_dict = {}
    for s in scores:
        logs_dict[s] = scores[s]
    print(logs_dict)

    #with open(osp.join(osp.dirname(opt.model_path), 'iou_pred_iter{}.pkl'.format(opt.eval_iter)), 'wb') as f:
    #    pickle.dump({'pred_ious':pred_ious, 'real_ious':real_ious}, f)

    eval_alarm_metrics(pred_ious, real_ious)

if __name__ == '__main__':
    main()
