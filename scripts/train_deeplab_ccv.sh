MODEL_NAME=$1
fold=$2
GPUs=$3

python tools/train_deeplab.py --name ${MODEL_NAME}_$fold --dataset_mode cityscapes \
                --dataroot ./cityscapes \
                --label_nc 19 --no_instance \
                --vgg_norm \
                --batchSize 8 \
                --lr 2e-4 \
                --n_fold 4 --fold $fold \
                --gpu_ids $GPUs \
                --niter 50000

