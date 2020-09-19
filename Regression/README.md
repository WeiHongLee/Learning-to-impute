# Learning to Impute
This is the Pytorch implementation on **Facial Landmark Regression** (i.e., Facial Keypoints Detection on the AFLW Dataset) of the paper 'Learning to Impute: A General Framework for Semi-supervised Learning' including code for the Supervised Learning (SL), Pseudo-labeling (PL) , Mean Teacher (MT) and our method L2I.


## Requirements
- Python 3.6+
- PyTorch 1.0 (or newer version)
- torchvision 0.2.2 (or newer version)
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Prepare Dataset
Download the dataset from https://github.com/tomasjakab/imm/. Unzip and place the whole dataset folder in ./data/

### Train Supervised Learning (SL) 
Train SL by full training data on AFLW dataset:
```
python train-SL.py --gpu <gpu_id> --n-labeled -1 --out aflwSL@-1
```
Train SL by 1% of labeled data on AFLW dataset:
```
python train-SL.py --gpu <gpu_id> --n-labeled 0.01 --out aflwSL@0.01
```

### Train Baselines
Train PL by 1% of labeled data on AFLW dataset:
```
python train-PL.py --gpu <gpu_id> --n-labeled 0.01 --out aflwPL@0.01
```

Train MT by 1% of labeled data on AFLW dataset:
```
python train-MT.py --gpu <gpu_id> --n-labeled 0.01 --out aflwMT@0.01
```

### Train Our L2I
*We provide code for L2I-O (modeling missing labels as the **O**utput of the model) and L2I-L (modeling missing labels as **L**earnable parameters) built on MT.*

Train our L2I-O built on MT by 1% labeled data on AFLW dataset:
```
python train-L2I-O.py --gpu <gpu_id> --n-labeled 0.01 --out aflwL2IO@0.01
```

Train our L2I-L built on MT by 1% labeled data on AFLW dataset:
```
python train-L2I-L.py --gpu <gpu_id> --n-labeled 0.01 --out aflwL2IL@0.01
```


### Monitoring training progress
```
tensorboard --port 6006 --logdir aflwL2IO@0.01
```

