MODEL_NAME=$1
GPUs=$2
python tools/train_fcn.py --name $MODEL_NAME --dataset_mode cityscapes \
                --dataroot ./cityscapes \
                --label_nc 19 --no_instance \
                --vgg_norm \
                --batchSize 8 \
                --lr 2e-4 \
                --n_fold 0 \
                --gpu_ids $GPUs \
                --snapshot 10000 \

