python train.py --name cityscapes --dataset_mode cityscapes \
                --dataroot ./datasets/cityscapes \
                --label_nc 19 --no_instance \
                --niter 50 \
                --batchSize 12 \
                --nThread 16 \
                --gpu_ids YOUR_GPU_IDS \
                --no_html --tf_log
