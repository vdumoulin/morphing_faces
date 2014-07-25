---
layout: default
---

# What is Morphing Faces?

Morphing Faces is an interactive Python demo allowing to generate images of
faces using a trained variational autoencoder.

The program maps a point in 400-dimensional space to an image and displays it on
screen. The point's position is initialized at random, and its coordinates can
be varied two dimensions at a time by hovering the mouse over the image.

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

# How does it work?

## Probabilistic graphical models and Bayesian networks

A probabilistic graphical model is a way to encode a distribution over random
variables as a graph, which can potentially yield a very compact representation
compared to a regular probability table. It does so by encoding dependences
between variables as edges between nodes.

Bayesian networks are a category of probabilistic graphical models whose
graphical representations are directed acyclic graphs (DAGs). The probability
distributions they encode are of the form

\\[
    P(X_1=x_1, X_2=x_2, \\cdots, X_n=x_n) = \\prod_{i=1}^n
        P(X_i=x_i \\mid \\mathcal{Pa}(X_i))
\\]

where \\( \\mathcal{Pa}(X_i) \\) is the set of \\( X_i \\)'s parents in the
graph.

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
  \\[
      p(\\mathbf{z}) = \\prod_{i} \\mathcal{N}(z_i \\mid 0, 1)
  \\]
* Isotropic gaussian approximate posterior distribution:
  \\[
      q(\\mathbf{z} \\mid \\mathbf{x}) = \\prod_{i}
          \\mathcal{N}(z_i \\mid \\mu(\\mathbf{x}), \\sigma^2(\\mathbf{x}))
  \\]
* Isotropic gaussian conditional distribution:
  \\[
      p(\\mathbf{x} \\mid \\mathbf{z}) = \\prod_{i}
          \\mathcal{N}(x_i \\mid \\mu(\\mathbf{z}), \\sigma^2(\\mathbf{z}))
  \\]
* Trained on the unlabel set of images of the
  [Toronto Face Database](http://aclab.ca/users/josh/TFD.html)
