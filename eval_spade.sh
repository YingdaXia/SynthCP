#python train.py --name GTA5_generator --dataset_mode cityscapes --label_dir ./datasets/GTA5/train_label -- image_dir ./datasets/GTA5/train_img
#python train.py --name GTA5_generator --dataset_mode custom --label_dir /data/gta5/labels --image_dir /data/gta5/images --no_instance --batchSize 32 --gpu_ids 0,1,2,3
python eval_ae.py --name cityscapes_dim256_vae --dataset_mode cityscapes \
                --dataroot ./cityscapes \
                --label_nc 19 --no_instance --continue_train --which_epoch 300 --phase test --no_flip \
                --batchSize 1 --serial_batches --use_vae --vae_test --eval_spade \
                --rec_save_suffix leftImg8bitRecVAE256E300\
                --nThread 16 \
                --gpu_ids 2 \


#--eval_spade
