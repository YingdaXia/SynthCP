EXP_NAME=fcn_exp
REC_PATH=leftImg8bitRec

# SynthCP-joint
IOUNET_NAME=synthcp_fcn_joint
IOUNET_MODEL_PATH=checkpoints/iounet/$IOUNET_NAME/iter20000.pth

# SynthCP-separate image-level
#IOUNET_NAME=synthcp_fcn_image-level 
#IOUNET_MODEL_PATH=checkpoints/iounet/$IOUNET_NAME/iter44000.pth

# SynthCP-separate pixel-level
#IOUNET_NAME=synthcp_fcn_pixel-level 
#IOUNET_MODEL_PATH=checkpoints/iounet/$IOUNET_NAME/iter28000.pth

# Direct Prediction
#IOUNET_NAME=direct_pred_fcn
#IOUNET_MODEL_PATH=checkpoints/iounet/$IOUNET_NAME/iter20000.pth

# prepare exp path
EXP_PATH=exps/$EXP_NAME
mkdir -p $EXP_PATH
mkdir -p $EXP_PATH/cityscapes
ln -s `pwd`/datasets/cityscapes/leftImg8bit $EXP_PATH/cityscapes/
ln -s `pwd`/datasets/cityscapes/gtFine $EXP_PATH/cityscapes/

# Run synthesize module to obtain reconstructions
# bash scripts/eval_spade.sh 2 $EXP_PATH 50 $REC_PATH train &
bash scripts/eval_spade.sh 3 $EXP_PATH 50 $REC_PATH test &
wait

# Train comparison module
# bash scripts/train_iounet.sh 0 $EXP_PATH $IOUNET_NAME $REC_PATH

# Evaluate comparison module
bash scripts/eval_iounet.sh 0 $EXP_PATH $IOUNET_MODEL_PATH $REC_PATH

####### IMPORTANT ###### For MSP, modify l78-81 in tools/eval_iounet_v2.py and run the code above again.