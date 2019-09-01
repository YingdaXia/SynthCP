"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import numpy as np
import torch
import os


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt, adda_mode = 'normal'):
        self.opt = opt
        self.adda_mode = adda_mode

        label_paths, image_paths, instance_paths = self.get_paths(opt, adda_mode = adda_mode)
        n_fold=opt.n_fold
        fold=opt.fold
        L = len(image_paths)
        if n_fold == 0:
            leaveout_indices = []
        else:
            leaveout_indices = np.arange(int(fold*L/n_fold), int((fold+1)*L/n_fold))

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)
            if opt.cross_validation_mode == 'train':
                instance_paths = [instance_paths[i] for i in range(L) if i not in leaveout_indices]
            else:
                instance_paths = [instance_paths[i] for i in range(L) if i in leaveout_indices]

        if opt.cross_validation_mode == 'train':
            label_paths = [label_paths[i] for i in range(L) if i not in leaveout_indices]
            image_paths = [image_paths[i] for i in range(L) if i not in leaveout_indices]
        else:
            label_paths = [label_paths[i] for i in range(L) if i in leaveout_indices]
            image_paths = [image_paths[i] for i in range(L) if i in leaveout_indices]

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.ignore_label = 255

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt, adda_mode = 'normal'):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)

        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, for_label=self.opt.eval_spade)
        label_tensor = transform_label(label) * 255.0


        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # for VGG segmentation network
        transform_image_vgg = get_transform(self.opt, params, for_VGG=True)
        image_tensor_vgg = transform_image_vgg(image)

        # label remapping
        if not self.opt.eval_spade:
            label_tensor_transform = self.ignore_label * torch.ones_like(label_tensor)
            for k,v in self.id_to_trainid.items():
                label_tensor_transform[label_tensor == k] = v
            label_tensor = label_tensor_transform
            label_tensor[label_tensor == self.ignore_label] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        if self.adda_mode == 'target': # target domain should not contain ground truth!
            input_dict = {
                        'instance': instance_tensor,
                        'image': image_tensor,
                        'image_seg':image_tensor_vgg,
                        'path': image_path,
                        }
        else:
            input_dict = {'label': label_tensor,
                        'instance': instance_tensor,
                        'image': image_tensor,
                        'image_seg':image_tensor_vgg,
                        'path': image_path,
                        'label_path': label_path,
                        }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
