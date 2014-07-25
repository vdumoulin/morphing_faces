---
layout: default
---

# What is Morphing Faces?

Morphing Faces is an interactive Python demo allowing to generate images of
faces using a trained variational autoencoder.

# How does it work?

## Variational autoencoders

First introduced by [(Kingma and Welling, 2014)](http://arxiv.org/abs/1312.6114)
and [(Rezende _et al._, 2014)](http://arxiv.org/abs/1401.4082), variational
autoencoders learn the parameters of a directed acyclic graph (DAG).

__[TODO: finish explanation]__

## Model specifications

* 400-dimensional latent space
* Encoding network has two hidden layers with 2000 rectified linear units each
* Decoding network has two hidden layers with 2000 rectified linear units each
* Isotropic gaussian prior distribution:
  \\[ p(\\mathbf{z}) = \\prod_{i} \\mathcal{N}(z_i \mid 0, 1) \\]

# Installation

## Dependencies

In addition to Python, Morphing Faces depends on the following Python packages:

* [numpy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/)

## Running the demo

To run the demo, simply download the code and type

```
python visualize.py
```

from within the `morphing_faces` directory.
