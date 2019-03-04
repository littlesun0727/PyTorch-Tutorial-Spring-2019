# PyTorch Tutorial
PyTorch tutorial codes for course EL-7143 Advanced Machine Learning, NYU, Spring 2019

## Requirements:
Python 3
Pytorch 1.01

## Notes:
We have 3 recitations in total:
1. Introduction to Neural Networks and PyTorch.
2. How to build a neural netowrk in PyTorch?
3. GANs, ResNets, Autoencoders, VisualBackProp, Skip connections, Batch normalization.

### Recitation 1:
1. calculation: ways to calculate cosine similarity of vectors.
2. classification: MNIST digits classification.
3. regression: regression for polynomial functions.

### Recitation 2:
Walk through PyTorch official examples: [MNIST](https://github.com/pytorch/examples/tree/master/mnist)

### Recitation 3:
1. autoencoder: autoencoder + Unet.
2. classification: STL10 image classification + ResNet + Visual BackProp.
3. GAN: generative adversarial nets.

## Requirments
Please install [PyTorch](http://pytorch.org/) as indicated. Please be careful about the *version* of Python, PyTorch and Cuda. I strongly recommend Python3 instead of Python2. Before you run the codes, check whether your machine supports GPU or not.

## Run
Run command ```python3 (name of python file).py```

The dataset should be downloaded automatically. `STL10` is a large dataset, and it may take several minutes.

## Homework Tips
1. You need to modify the codes and add functions like plotting for your homework. 
2. Take care of regression part, because you need to deal with two variables.
3. Read option.py since you may need to adjust the parameters.

### About
The recitations were prepared all by your beloved teaching assistant. He is an ordinary student just like you and this is a volunteer work for him. 

Theory and implementation can always be different. If there are any conflicts between the recitations and the lectures, please use your **lecture notes** as a reference. e.g. you may find GAN was implemented a little bit different in the codes compared with Ian's paper and the lecture notes. You'd better write the algorithms in the original paper instead of coping and pasting my codes in the midterm or final.

## Thanks
All the codes are inspired by [PyTorch official examples](https://github.com/pytorch/examples). Also thanks my professor's trust to let me take the responsibility to teach this recitation.
