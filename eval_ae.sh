#python train.py --name GTA5_generator --dataset_mode cityscapes --label_dir ./datasets/GTA5/train_label -- image_dir ./datasets/GTA5/train_img
#python train.py --name GTA5_generator --dataset_mode custom --label_dir /data/gta5/labels --image_dir /data/gta5/images --no_instance --batchSize 32 --gpu_ids 0,1,2,3
python eval_ae.py --name cityscapes_dim8_c19 --dataset_mode cityscapes \
                --dataroot /data/yzhang/cityscapes \
                --label_dir /data/yzhang/gta5_deeplab/labels \
                --image_dir /data/yzhang/gta5_deeplab/images \
                --label_nc 19 --no_instance --continue_train --which_epoch 50 --eval_losses_dir . --phase test \
                --batchSize 1 --serial_batches \
                --nThread 16 \
                --gpu_ids 0 \
                --use_vae \
                --z_dim 8 \

