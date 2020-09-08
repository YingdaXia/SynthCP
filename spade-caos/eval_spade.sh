python eval_ae.py --name caos --dataset_mode caos \
                --dataroot ./anomaly/data \
                --label_nc 13 --no_instance --continue_train --which_epoch 200 --phase test --no_flip \
                --batchSize 1 --serial_batches\
                --label_dir ../anomaly/data/test_result \
                --rec_save_path ../anomaly/data/test_recon\
                --eval_spade \
                --nThread 16 \
                --gpu_ids 0 \
