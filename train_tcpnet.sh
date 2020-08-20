GPUs=$1
eval_output_dir=$2
IOUNET_NAME=$3
#cityscapes_iouconf_hce2_1_concatInput_concatNet
python train_tcpnet.py --name $IOUNET_NAME --dataset_mode iou \
                --dataroot ./$eval_output_dir/cityscapes \
                --image_src_dir ./$eval_output_dir/cityscapes/leftImg8bitResize/train \
                --image_rec_dir ./$eval_output_dir/cityscapes/leftImg8bitRec/train \
                --iou_dir ./$eval_output_dir/metrics_trainccv \
                --pred_dir ./$eval_output_dir/cityscapes/gtFinePredProb/train \
                --model_path checkpoints/iounet/tcp_fcn/iter40000.pth \
                --label_nc 19 --no_instance \
                --use_vae \
                --vgg_norm \
                --batchSize 4 \
                --lr 5e-5 \
                --niter 100000 \
                --snapshot 5000 \
                --gpu_ids $GPUs --nThreads 8

#                --model_path checkpoints/fcn8s/cityscapes_c19-iter100000.pth \
#                --model_path checkpoints/deeplab/cityscapes_c19-iter50000.pth \
