# Reproducing the result of TCP. Assume the segmentation network has been run.
EXP_NAME=fcn_exp
SEG_MODEL_NAME=cityscapes_c19
EXP_PATH=exps/$EXP_NAME

###### IMPORTANT #######
# YOU NEED TO MODIFY LINE 35-37 IN tools/eval_tcpnet.py TO USE THE DESIRED MODEL
TCPNET_NAME=tcpnet_fcn
TCPNET_MODEL_PATH=checkpoints/iounet/$TCPNET_NAME/iter40000.pth

# TCPNET_NAME=tcpnet_deeplab
# TCPNET_MODEL_PATH=checkpoints/iounet/$TCPNET_NAME/iter30000.pth

mkdir -p $EXP_PATH
mkdir -p $EXP_PATH/cityscapes
ln -s `pwd`/datasets/cityscapes/leftImg8bit $EXP_PATH/cityscapes/
ln -s `pwd`/datasets/cityscapes/gtFine $EXP_PATH/cityscapes/

# Pixel-level failure detection
#bash scripts/train_tcpnet.sh $SEG_MODEL_NAME 0 $EXP_PATH $TCPNET_NAME
bash scripts/eval_tcpnet.sh 0 $EXP_PATH $TCPNET_MODEL_PATH

