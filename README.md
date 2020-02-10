
<br><br><br>

# CycleGAN in PyTorch
We provide PyTorch implementations for both unpaired and paired image-to-image translation.
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)

The code was strongly inspired by the code written by : [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung).


<img src="https://github.com/AissamDjahnine/cycle/blob/master/imgs/head.jpg" width="800"/>



## Motivation


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  
  
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.



## Acknowledgments
Our code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).

### Tensorboard 

If you're familiar with Tensorboard , skip this section 

In your terminal, run:

```bash
tensorboard --logdir ./runs
```

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

