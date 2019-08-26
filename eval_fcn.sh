python eval.py --name cityscapes_from_GTA5 --dataset_mode cityscapes \
                --phase val \
                --label_nc 35 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 8 \
                --gpu_ids 7