GPUs=$1
eval_output_dir=$2
EPOCH=$3
PHASE=$4
#python eval_ae.py --name cityscapes_label_only_c19 --dataset_mode cityscapes \
python eval_ae.py --name cityscapes_dim256_vae --dataset_mode cityscapes \
                --dataroot ./$eval_output_dir/cityscapes \
                --label_nc 19 --no_instance --continue_train --which_epoch $EPOCH --phase $PHASE --no_flip \
                --batchSize 1 --serial_batches --eval_spade --use_vae --vae_test \
                --rec_save_suffix leftImg8bitRecVAE256E200 \
                --nThread 16 \
                --gpu_ids $GPUs \
                --eval_output_dir $eval_output_dir \


#--eval_spade
