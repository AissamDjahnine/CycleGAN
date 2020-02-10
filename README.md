
<br><br><br>

# CycleGAN in PyTorch
We provide PyTorch implementation for both unpaired and paired image-to-image translation applied for medical image segmentation.
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)

The code was strongly inspired by the code of : [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung).

<img src="https://github.com/AissamDjahnine/cycle/blob/master/imgs/head.jpg" width="800"/>

## Motivation
### The main objective of this work is twofold :
* Generate synthetic cell images that model the distribution
of the input images for data augmentation. Use both of
the synthetic and real cells images for training a contextaware CNN that can accurately segment these cells. See : [GANs](https://github.com/AissamDjahnine/gans)

* We propose to employ a segmentation method based on cycle-consistent generative adversarial networks that can be trained even in absence of prepared image-mask pairs

## Prerequisites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/AissamDjahnine/CycleGAN.git
cd CycleGAN
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  
  
- Train a model:
```bash
python train.py --dataroot ./datasets/rpedata --name rpedata --model cycle_gan --gan_mode vanilla --dataset_mode unaligned --n_epochs 100 --n_epochs_decay 50 --save_epoch_freq 20 
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:

* Change the `--dataroot` and `--name` to be consistent with your trained model's configuration.

* Using --model cycle_gan requires loading and generating results in both directions. The results will be saved at ./results/. Use --results_dir {directory_path_to_save_result} to specify the results directory.

* For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.

```bash
python test.py --dataroot datasets/rpedata --name rpedata --dataset_mode aligned --no_dropout  --model cycle_gan --norm
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

## Unpaired, Paired datasets : 

* For paired mode, please use the python function : combine_A_B.py to join train images. you should have two folders : train and test with joined images.

* For unpaired mode, you should have 4 folders : trainA, trainB, testA, testB with images in both domains.


* You can download an unpaired version of Human U2OS cells : [Unpaired_dataset](https://drive.google.com/file/d/1qmNF0GBrR8OP7s4zzZHaVwXUvSbKDsJX/view?usp=sharing). 

* You can download a paired version of Human Retinal Pigment Epithelium (RPE) cells : [Paired data](https://drive.google.com/file/d/1dwfrabUz5WsPEmVqDrZrtZZm41ccutR8/view?usp=sharing)


## Notebook :
* You find the notebook associated with this projects. after cloning the project , open the notebook and follow instructions

## Tensorboard 

If you're familiar with Tensorboard , skip this section 

In your terminal, run:

```bash
tensorboard --logdir ./runs
```

* You should be looking to :
![gans](https://github.com/AissamDjahnine/CycleGAN/blob/master/imgs/tensorboard.jpg)

## Acknowledgments
Our code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
## References 

* Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\,  [Taesung Park](https://taesung.me/)\, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017.

* "Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.

* "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" Alec Radford, Luke Metz, Soumith Chintala

## Questions :
[![Generic badge](https://img.shields.io/badge/TEST-VERSION-RED.svg)](https://github.com/AissamDjahnine)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/AissamDjahnine)
[![GitHub followers](https://img.shields.io/github/followers/Naereen.svg?style=social&label=Follow&maxAge=2592000)](https://github.com/AissamDjahnine?tab=followers)

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please feel free to contact if case of issues.

