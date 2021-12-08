# D2M-GAN: music generation for dance videos.
This is the official implementation for **D2M-GAN**, a VQ-GAN based music generative model for dance videos.
<!-- [arXiv]() | [Project Page]() | [Bibtex](#bibtex)  -->


## 0. Project Overview
In our paper Quantized GAN for Complex Music Generation from Dance Videos, we present Dance2Music-GAN (D2M-GAN), a novel adversarial multi-modal framework that generates complex musical samples conditioned on dance videos.

<p align="center">
	<img src="assets/d2m.png" width="700">


## 1. Environment Setup

The envirunment can be set up following the instructions below.
Since our proposed **D2M-GAN** utilizes the VQ-VAE decoder of JukeBox for synthesizing the music, part of the code for loading the pre-trained VQ-VAE are adapted from the [JukeBox repo](https://github.com/openai/jukebox).

 
```
conda create --name d2m python=3.8
source activate d2m
conda install mpi4py==3.0.3
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
git clone https://github.com/L-YeZhu/D2M-GAN.git
cd D2M-GAN
pip install -r requirements.txt
pip install -e .
```


## 2. Data

We conduct experiments on the following two datasets.

### 2.1 AIST++ Dataset
The AIST++ dataset is a subset of AIST dataset, which can be downloaded from .
We use the cross-modality data split for training and testing. 


### 2.2 TikTok Dance-Music Dataset
We build a TikTok Dance-Music dataset, which contains the "in-the-wild" dance videos. The current dataset has 445 dance videos with an average length of 12.5 seconds. This dataset can be downloaded from our project page.




## 3. Training

### 3.1 Training w/ AIST++
To train the D2M-GAN on 

#python music_gan_top.py --model=5b --name=sample_5b --levels=3 --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125

```
python 
```

### 3.2 Training w/ TikTok Dance-Music

## 4. Generation

## 5. Project Page and Pre-trained models
For qualitative examples of generated music, please refer to our project page.
We also provide 

## 6. Citation
Please consider citing our paper if you find it useful.

