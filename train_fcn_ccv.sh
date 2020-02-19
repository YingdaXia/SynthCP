fold=$1
GPUs=$2
python train_fcn.py --name cityscapes_c19_$fold --dataset_mode cityscapes \
                --dataroot ./cityscapes \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 8 \
                --lr 2e-4 \
                --n_fold 4 --fold $fold \
                --gpu_ids $GPUs \
                --snapshot 10000 \
#                --load_size 1024 \
#                --crop_size 1024

