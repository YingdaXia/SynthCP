python train.py --name caos --dataset_mode caos \
                --dataroot ../anomaly/data/ \
                --label_nc 13 --no_instance \
                --niter 100 --niter_decay 100 \
                --batchSize 16 \
                --nThread 15 \
                --gpu_ids 0,1,2,3 \
                --no_html --tf_log
