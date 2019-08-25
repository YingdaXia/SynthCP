#python train.py --name GTA5_generator --dataset_mode cityscapes --label_dir ./datasets/GTA5/train_label -- image_dir ./datasets/GTA5/train_img
#python train.py --name GTA5_generator --dataset_mode custom --label_dir /data/gta5/labels --image_dir /data/gta5/images --no_instance --batchSize 32 --gpu_ids 0,1,2,3
python train.py --name cityscapes_from_GTA5_image_encoder --dataset_mode custom \
                --label_dir /data/yzhang/gta5_deeplab/labels \
                --image_dir /data/yzhang/gta5_deeplab/images \
                --label_nc 35 --no_instance \
                --use_vae \
                --batchSize 6 \
                --gpu_ids 6,7 \
                --continue_train