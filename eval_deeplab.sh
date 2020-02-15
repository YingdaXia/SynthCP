python eval_deeplab.py --name cityscapes_c19 --dataset_mode cityscapes \
                --phase test \
                --label_nc 19 --no_instance --no_flip \
                --dataroot ./cityscapes \
                --use_vae \
                --vgg_norm \
                --model_path ./checkpoints/deeplab/cityscapes_c19-iter40000.pth \
                --gpu_ids 3 \
                --load_size 1024 \
                --crop_size 1024
#--model_path checkpoints/fcn8s/cityscapes_from_GTA5-iter100000.pth \
