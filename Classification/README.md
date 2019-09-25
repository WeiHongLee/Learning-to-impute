# Learning to Label
This is the Pytorch implementation on **classification** of the ICLR paper 'Learning to Label for Semi-Supervised Learning' including code for MixMatch, Mean Teacher (MT) and both the option1 and option2 proposed in 'Learning to Label for Semi-Supervised Learning'.


## Requirements
- Python 3.6+
- PyTorch 1.0 (or newer version)
- torchvision 0.2.2 (or newer version)
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Train
Train the option1 by 250 labeled data of CIFAR-10 dataset (around 4 days on 1 1080Ti):

```
python train-Ours-option1.py --gpu <gpu_id> --n-labeled 250 --out cifar10Opt1@250
```

### Monitoring training progress
```
tensorboard.sh --port 6006 --logdir cifar10Opt1@250
```

