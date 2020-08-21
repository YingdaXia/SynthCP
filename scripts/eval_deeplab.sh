python tools/eval_deeplab.py --name cityscapes_c19 --dataset_mode cityscapes \
                --phase test \
                --label_nc 19 --no_instance --no_flip \
                --dataroot ./cityscapes \
                --vgg_norm \
                --model_path ./checkpoints/deeplab/cityscapes_c19-iter50000.pth \
                --gpu_ids 0 \
