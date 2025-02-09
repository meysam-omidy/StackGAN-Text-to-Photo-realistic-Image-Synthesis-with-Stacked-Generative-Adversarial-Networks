## Overview
This repository contains the implementation of the paper "StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks" by Han Zhang et al ([link to the paper](https://arxiv.org/abs/1612.03242)). StackGAN is a framework that generates high-resolution, photo-realistic images from text descriptions. It decomposes the problem into two stages:

1. **Stage-I GAN**: Generates low-resolution images with rough shapes and basic colors of the object from the text description.
2. **Stage-II GAN**: Refines the Stage-I results by adding more details and correcting defects to produce high-resolution images.

## Features

- **Conditioning Augmentation**: Enhances diversity in generated images and stabilizes GAN training.
- **Two-stage Architecture**: Improves image quality by sequentially refining the initial sketches.

## Usage
1. Download main images dataset from ([here](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images))
2. Download text descriptions from ([here](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE/view?resourcekey=0-sZrhftoEfdvHq6MweAeCjA))
3. Download includes.txt file for matching images to texts from ([here](https://drive.google.com/file/d/1X5YArm5ZUtB_37G9cln7e9NWqXG9uKty/view?usp=sharing))
4. After extracting all files, you should have following structure:
```
.models.py
.stage1_train.py
.stage2_train.py
.utis.py
.includes.txt
\CUB_200_2011
\birds
```
5. Train stage1_generator and stage1_discriminator using stage1_train.py
6. Train stage2_generator and stage2_discriminator using stage2_train.py, by passing pre-trained stage1_generator to stage2_generator

## References

- Original paper: [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242)
- Official code: [GitHub Repository](https://github.com/hanzhanggit/StackGAN)
