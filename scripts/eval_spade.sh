GPUs=$1
eval_output_dir=$2
EPOCH=$3
REC_PATH=$4
PHASE=$5
python tools/eval_ae.py --name cityscapes_label_only_c19 --dataset_mode cityscapes \
                --dataroot ./$eval_output_dir/cityscapes \
                --label_nc 19 --no_instance --continue_train --which_epoch $EPOCH --phase $PHASE --no_flip \
                --batchSize 1 --serial_batches --eval_spade  --vae_test \
                --rec_save_suffix $REC_PATH \
                --nThread 16 \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \
