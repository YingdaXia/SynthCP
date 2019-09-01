"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from PIL import Image
from util.util import tensor2im
import torch
import os
import pdb

#torch.backends.cudnn.benchmark = False

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)
epoch = 0
iter_counter.record_epoch_start(epoch)
for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
    iter_counter.record_one_iteration()

    # Training
    # train generator
    with torch.no_grad():
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, backward=False)

        # train discriminator
        trainer.run_discriminator_one_step(data_i, backward=False)

    # Visualizations
    losses = trainer.get_latest_losses()
    visualizer.record_losses(iter_counter.epoch_iter, losses)

    # Save reconstructed image
    img_rec_path = data_i['path'][0].replace('leftImg8bit', 'leftImg8bitRec')
    os.makedirs(os.path.dirname(img_rec_path), exist_ok=True)
    Image.fromarray(tensor2im(trainer.get_latest_generated()[0])).save(img_rec_path)

    visuals = OrderedDict([('input_label', data_i['label']),
                           ('synthesized_image', trainer.get_latest_generated()),
                           ('real_image', data_i['image'])])
    visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

visualizer.dump_record_losses()
print('Evaluating was successfully finished.')
