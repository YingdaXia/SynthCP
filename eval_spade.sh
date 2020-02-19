GPUs=$1
eval_output_dir=$2
EPOCH=$3
PHASE=$4
python eval_ae.py --name cityscapes_new --dataset_mode cityscapes \
                --dataroot ./cityscapes \
                --label_nc 19 --no_instance --continue_train --which_epoch $EPOCH --phase $PHASE --no_flip \
                --batchSize 1 --serial_batches --eval_spade \
                --rec_save_suffix leftImg8bitRec \
                --nThread 16 \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \


#--eval_spade
