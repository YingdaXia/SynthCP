python train.py --name caos --dataset_mode caos \
                --dataroot /PATH/TO/DATAROOT \
                --label_nc 13 --no_instance \
                --niter 100 --niter_decay 100 \
                --batchSize 12 \
                --nThread 15 \
                --gpu_ids YOUR_GPU_IDs \
                --no_html --tf_log
