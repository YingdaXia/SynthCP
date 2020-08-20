#EXP_NAME=deeplab_lr
EXP_NAME=fcn_lr

#LINREG_NAME=cityscapes_iouconf_mcdropout_dl
LINREG_NAME=cityscapes_iouconf_mcdropout_reproduce
#bash train_fcn.sh 1
#bash train_fcn_ccv.sh 0 0 &
#bash train_fcn_ccv.sh 1 1 &
#bash train_fcn_ccv.sh 2 2 &
#bash train_fcn_ccv.sh 3 3 &
#wait
#bash eval_fcn_mcdropout.sh 0 0 $EXP_NAME &
#bash eval_fcn_mcdropout.sh 1 1 $EXP_NAME &
#bash eval_fcn_mcdropout.sh 2 2 $EXP_NAME &
#bash eval_fcn_mcdropout.sh 3 3 $EXP_NAME &
#wait
#bash eval_fcn_mcdropout_test.sh 0 $EXP_NAME

#bash eval_deeplab_mcdropout.sh 0 0 $EXP_NAME &
#bash eval_deeplab_mcdropout.sh 1 1 $EXP_NAME &
#wait
#bash eval_deeplab_mcdropout.sh 2 0 $EXP_NAME &
#bash eval_deeplab_mcdropout.sh 3 1 $EXP_NAME &
#wait
#bash eval_deeplab_mcdropout_test.sh 0 $EXP_NAME
#
#
#bash train_linreg.sh 0 $EXP_NAME $LINREG_NAME
#bash eval_linreg.sh 0 $EXP_NAME $LINREG_NAME
bash eval_mcdropout.sh 0 $EXP_NAME

