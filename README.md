# Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation

### [Project page](https://) |   [Paper](https://arxiv.org/pdf/2003.08440.pdf) | [Video](https://) 

Synthesize then Compare: Detecting Failures and Anomalies for Semantic Segmentation <br>
[Yingda Xia*](https://yingdaxia.xyz),  [Yi Zhang*](https://edz-o.github.io), [Fengze Liu](https://scholar.google.com/citations?user=T3EjsaAAAAAJ&hl=en), [Wei Shen](http://wei-shen.weebly.com/), and [Alan Yuille](https://www.cs.jhu.edu/~ayuille/).<br>
In ECCV 2020 (Oral).

## Installation

Clone this repo.
```bash
git clone https://github.com/XXXXX/XXXXX.git
cd XXXXX/
```

This code has been tested with PyTorch 1.2.0 and python 3.7. Please install dependencies by
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download Cityscapes [dataset](https://www.cityscapes-dataset.com/) and place it under `datasets/`

## Running the code

First, download our checkpoint models from [here](#) and name the folder as `checkpoints/'. You should have a structure like this,

```
checkpoints/
  |----fcn8s/
  |----deeplab/
  |----iounet/
  |----cityscapes_label_only_c19/
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

To train SynthCP, you need to first train the segmentation models using cross-validation on Cityscapes training set, 

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
This code borrows heavily from SPADE. 
