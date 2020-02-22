
import torch.utils.data as data
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform
import util.util as util
import numpy as np
import random
import json
import pdb

from data.image_folder import make_dataset, make_iou_dataset, is_npz_file

class IOUEntropyDataset(BaseDataset):

    def initialize(self, opt):
        self.with_conf_map = False
        image_src_paths, image_rec_paths, label_paths, pred_paths, entropy_paths, conf_map_paths = self.get_paths(opt)
        util.natural_sort(image_src_paths)
        util.natural_sort(image_rec_paths)
        util.natural_sort(label_paths)
        util.natural_sort(pred_paths)
        util.natural_sort(entropy_paths)
        if opt.with_conf_map:
            self.with_conf_map = opt.with_conf_map
            util.natural_sort(conf_map_paths)
            self.conf_map_paths = conf_map_paths

        self.image_src_paths = image_src_paths
        self.image_rec_paths = image_rec_paths
        self.label_paths = label_paths
        self.entropy_paths = entropy_paths
        self.pred_paths = pred_paths
        print(len(label_paths))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ])


    def __getitem__(self, index):

        # input image (real images)
        image_src_path = self.image_src_paths[index]
        image_rec_path = self.image_rec_paths[index]
        label_path = self.label_paths[index]
        entropy_path = self.entropy_paths[index]
        if self.with_conf_map:
            conf_map_path = self.conf_map_paths[index]
            conf_map = np.load(conf_map_path)['confidence_map']
        else:
            conf_map = 0
        pred_path = self.pred_paths[index] + '.npz'
        assert self.paths_match(label_path, image_src_path, image_rec_path), \
            "The label_path %s, image_src_path %s and image_rec_path %s don't match." % \
            (label_path, image_src_path, image_rec_path)

        image_src = Image.open(image_src_path).convert('RGB')
        image_rec = Image.open(image_rec_path).convert('RGB')
        iou_label = json.load(open(label_path, 'r'))
        entropy = json.load(open(entropy_path, 'r'))

        prob_map, label_map = np.load(pred_path)['prob'], np.load(pred_path)['label']
        prob_map = torch.from_numpy(prob_map)
        label_map = torch.from_numpy(label_map)

        image_src_tensor = self.transform(image_src)
        image_rec_tensor = self.transform(image_rec)

        data = {
                  'image_src' : image_src_tensor,
                  'image_rec' : image_rec_tensor,
                  'iou' : torch.tensor( iou_label[0]),
                  'valid' : torch.tensor(iou_label[1]) != 0,
                  'entropy' : torch.tensor(entropy[5]),
                  'image_src_path' : image_src_path,
                  'prob' : prob_map,
                  'label_map' : label_map,
                  'conf_map' : conf_map
                }
        return data

    def __len__(self):
        return len(self.image_src_paths)

    def get_paths(self, opt):

        image_src_paths = make_dataset(opt.image_src_dir, recursive=True)
        image_rec_paths = make_dataset(opt.image_rec_dir, recursive=True)
        label_paths = make_iou_dataset(opt.iou_dir, recursive=True)
        entropy_paths = make_iou_dataset(opt.entropy_dir, recursive=True)
        pred_paths = make_dataset(opt.pred_dir, recursive=True)
        if opt.with_conf_map:
            conf_map_paths = make_dataset(opt.conf_map_dir, recursive=True, is_target_file=is_npz_file)
        else:
            conf_map_paths = None

        return image_src_paths, image_rec_paths, label_paths, pred_paths, entropy_paths, conf_map_paths

    def paths_match(self, path1, path2, path3):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        name3 = os.path.basename(path3)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3]) and \
            '_'.join(name3.split('_')[:3])



