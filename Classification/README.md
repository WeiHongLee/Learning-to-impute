# Learning to Impute
This is the Pytorch implementation on **Image Classification** on CIFAR-10 of the paper 'Learning to Impute: A General Framework for Semi-supervised Learning' including code for MixMatch (MM, Code is adapted from [here](https://github.com/YU1ut/MixMatch-pytorch)), Mean Teacher (MT) and our method L2I.


## Requirements
- Python 3.6+
- PyTorch 1.0 (or newer version)
- torchvision 0.2.2 (or newer version)
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Train Baselines
Train the MT by 250 labeled data of CIFAR-10 dataset:
```
python train-MT.py --gpu <gpu_id> --n-labeled 250 --out cifar10MT@250
```

Train the MM by 250 labeled data of CIFAR-10 dataset:
```
python train-MM.py --gpu <gpu_id> --n-labeled 250 --out cifar10MM@250
```

### Train Our L2I
*We provide code for L2I-O (modeling missing labels as the **O**utput of the model) and L2I-L (modeling missing labels as **L**earnable parameters) built on MT.*

Train our L2I-O built on MT by 250 labeled data of CIFAR-10 dataset:
```
python train-L2I-O.py --gpu <gpu_id> --n-labeled 250 --out cifar10L2IO@250
```

Train our L2I-L built on MT by 250 labeled data of CIFAR-10 dataset:
```
python train-L2I-L.py --gpu <gpu_id> --n-labeled 250 --out cifar10L2IL@250
```

### Monitoring training progress
```
tensorboard --port 6006 --logdir cifar10L2IO@250
```

