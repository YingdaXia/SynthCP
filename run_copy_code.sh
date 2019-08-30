EXP_NAME=$1
GPUs=$2

mkdir $EXP_NAME
cp -r models $EXP_NAME
cp -r datasets $EXP_NAME
cp -r trainers $EXP_NAME
cp -r data $EXP_NAME
cp -r util $EXP_NAME
cp -r options $EXP_NAME
cp *.py $EXP_NAME
cp *.sh $EXP_NAME

cd $EXP_NAME

BATCH_SIZE=16
RUN=False
if [ $RUN = True ]
then
python train.py --name cityscapes_dim8_c19 --dataset_mode cityscapes \
                --dataroot /data/yzhang/cityscapes \
                --label_dir /data/yzhang/gta5_deeplab/labels \
                --image_dir /data/yzhang/gta5_deeplab/images \
                --label_nc 19 --no_instance \
                --batchSize $BATCH_SIZE \
                --nThread 16 \
                --gpu_ids $GPUs \
                --use_vae \
                --z_dim 8
fi

