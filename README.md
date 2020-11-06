# Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation

### [Project page](#) |   [Paper](https://arxiv.org/pdf/2003.08440.pdf) | [Video](#) 

Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation <br>
[Yingda Xia*](https://yingdaxia.xyz),  [Yi Zhang*](https://edz-o.github.io), [Fengze Liu](https://scholar.google.com/citations?user=T3EjsaAAAAAJ&hl=en), [Wei Shen](http://wei-shen.weebly.com/), and [Alan Yuille](https://www.cs.jhu.edu/~ayuille/).<br>
In ECCV 2020 (Oral).

## Installation

Clone this repo.
```bash
git clone https://github.com/YingdaXia/SynthCP.git
cd SynthCP
```

This code has been tested with PyTorch 1.2.0 and python 3.7. Please install dependencies by
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download Cityscapes [dataset](https://www.cityscapes-dataset.com/) and place it under `datasets/`.

Download StreetHards dataset following this [repo](https://github.com/hendrycks/anomaly-seg). Please download both train and test data, and arrange them as ```anomaly/data/train``` and ```anomaly/data/test``` respectively.

## Running the code on the Cityscpaes

First, download our checkpoint models from [here](https://www.cs.jhu.edu/~yzh/synthcp_failure_checkpoints.zip) and name the folder as `checkpoints/'. You should have a structure like this,

```
checkpoints/
  |----fcn8s/
  |----deeplab/
  |----iounet/
  |----cityscapes_label_only_c19/
  |----caos/
  |----caos-segmentation/
```

### Obtaining paper results
Take FCN-8s as example, you will need to test the FCN model on the testset of Cityscapes. See `scripts/eval_segmentation_models.sh`.

```bash
bash scripts/eval_segmentation_models.sh
```

Then run the scripts we provided to obtain numbers. 

- SynthCP: `scripts/reproduce_synthcp_{fcn,deeplab}.sh`
- Direct Prediction: `scripts/reproduce_synthcp_{fcn,deeplab}.sh`
- MSP: `scripts/reproduce_synthcp_{fcn,deeplab}.sh` 
- MCDropout: `scripts/reproduce_mcdropout_{fcn,deeplab}.sh`
- TCP: `scripts/reproduce_tcp.sh`

### Training 

To train the synthesize module (SPADE), use the following script:
```bash
cd spade-cityscapes
bash run.sh
```
We provided our pre-trained SPADE (on the label-image pairs) in ```checkpoints/cityscapes_label_only_c19/```.

To train the comparison module, you need to first train the segmentation models using cross-validation on Cityscapes training set, 

```bash
bash scripts/train_segmentation_models.sh
```

Then evaluate the model on the validation set of each fold (See the commented lines in `scripts/eval_segmentation_models.sh`).

```bash
bash scripts/eval_segmentation_models.sh
```

Also run the synthesize module on the training set (See the commented lines in `scripts/reproduce_synthcp_{fcn,deeplab}.sh`). Train the comparison module,

```bash
bash scripts/train_iounet.sh 0 $EXP_PATH $IOUNET_NAME $REC_PATH
```

Similar process for Direct Prediction, image-level MCDropout and TCP.

## Running the code on StreetHazards dataset

First, train and test segmentation model to obtain segmentation predictions (saved in ```anomaly/data/test_result``` by default).
```
cd anomaly
python train.py
python test.py
```
We provided our trained segmentation model in ```checkpoints/caos-segmentation```. 

Then, train the synthesize module (SPADE). 
```
cd ../spade-caos
bash run.sh
```
And use it to obtain reconstructions of the segmentation predictions (saved in ```anomaly/data/test_recon``` by default).
```bash
bash eval_spade.sh
```
We also provided our trained GAN model in ```checkpoints/caos```. If you want to use it, please copy it to ```spade-caos/checkpoints/caos```.

Finally, segment anomaly objects by computing a feature-space distance between the images and the reconstructions.
```
cd ../anomaly
python eval_ood_rec.py
```

### Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{xia2020synthesize,
  title={Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation},
  author={Xia, Yingda and Zhang, Yi and Liu, Fengze and Shen, Wei and Yuille, Alan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Acknowledgments
The code for training and testing the synthesize module is extended from [SPADE](https://github.com/nvlabs/spade/) (Copyright (C) 2019 NVIDIA Corporation). The code for anomaly segmentation is extended from [anomaly-seg](https://github.com/hendrycks/anomaly-seg), which is also built on [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch).
