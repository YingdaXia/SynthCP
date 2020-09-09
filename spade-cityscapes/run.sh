python train.py --name cityscapes --dataset_mode cityscapes \
                --dataroot ../datasets/cityscapes \
                --label_nc 19 --no_instance \
                --niter 50 \
                --batchSize 12 \
                --nThread 16 \
                --gpu_ids 0,1,2,3 \
                --no_html --tf_log
