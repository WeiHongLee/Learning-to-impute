# Learning to Impute
This is the Pytorch implementation on **Regression** (Facial Keypoints Detection on the AFLW Dataset) of the paper 'Learning to Impute: A General Framework for Semi-supervised Learning' including code for the Supervised Learning (SL), Pseudo Label (PL) , Mean Teacher (MT) and both the option1 and option2 proposed in 'Learning to Impute: A General Framework for Semi-supervised Learning'.


## Requirements
- Python 3.6+
- PyTorch 1.0 (or newer version)
- torchvision 0.2.2 (or newer version)
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Download the Dataset
Download the dataset from https://github.com/tomasjakab/imm/. Unzip and place the whole dataset folder in ./data/

### Train
Train the option1 by 1% of labeled data of AFLW dataset:

```
python train-Ours-option1.py --gpu <gpu_id> --n-labeled 0.01 --out alfwOpt1@0.01
```

### Monitoring training progress
```
tensorboard --port 6006 --logdir alfwOpt1@0.01
```

