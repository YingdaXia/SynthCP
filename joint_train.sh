python joint_train.py --name cityscapes_from_GTA5_joint_train \
                --init_name cityscapes_from_GTA5_image_encoder \
                --dataset_mode custom \
                --label_dir /data/yzhang/gta5_deeplab/labels \
                --image_dir /data/yzhang/gta5_deeplab/images \
                --label_nc 35 --no_instance \
                --use_vae \
                --batchSize 1 \
                --joint_train \
                --gpu_ids 6