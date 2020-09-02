python train.py --name cityscapes --dataset_mode cityscapes \
                --dataroot /PATH/TO/CITYSCAPES \
                --label_nc 19 --no_instance \
                --niter 50 \
                --batchSize 12 \
                --nThread 16 \
                --gpu_ids 2,4,7 \
                --no_html --tf_log
