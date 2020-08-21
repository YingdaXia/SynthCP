python tools/eval_fcn.py --name cityscapes_c19 --dataset_mode cityscapes \
                --phase test \
                --label_nc 19 --no_instance --no_flip \
                --dataroot ./cityscapes \
                --use_vae \
                --vgg_norm \
                --model_path ./checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
                --gpu_ids 0 \
