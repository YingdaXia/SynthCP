python fcn_plus_iounet.py --name joint_eval --dataset_mode cityscapes \
                --phase test \
                --label_nc 19 --no_instance --no_flip --serial_batches\
                --dataroot /data/yzhang/cityscapes \
                --fcn_model_path /data/yingda/Domain-Adaptation/checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
                --SPADE_model_path /data/yzhang/Domain-Adaptation/checkpoints/cityscapes_label_only_c19/50_net_G.pth \
                --IOUNet_model_path /data/yzhang/Domain-Adaptation/checkpoints/fcn8s/cityscapes_iounet_c19-iter50000.pth \
                --gpu_ids 1
#--model_path checkpoints/fcn8s/cityscapes_from_GTA5-iter100000.pth \
