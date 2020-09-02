python eval_ae.py --name caos --dataset_mode caos \
                --dataroot /PATH/TO/DATAROOT \
                --label_nc 13 --no_instance --continue_train --which_epoch 200 --phase test --no_flip \
                --batchSize 1 --serial_batches\
                --label_dir /PATH/TO/TEST/RESULTS \
                --rec_save_path /PATH/TO/RECONSTRUCTION\
                --eval_spade \
                --nThread 16 \
                --gpu_ids [GPU_IDS] \
