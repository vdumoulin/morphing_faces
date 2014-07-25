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

This section supposes a background in probabilities and recapitulates basic
concepts of probabilistic graphical models before introducing variational
autoencoders. For a (much) more thorough treatment of probabilistic graphical
models, see the excellent
[Probabilistic Graphical Models - Principles and Techniques](http://pgm.stanford.edu/)
textbook written by Daphne Koller and Nir Friedman. The introductory material
assumes that all variables are discrete for the sake of simplicity, although
everything discussed also applies for continuous variables if sums are replaced
by integrals where appropriate.

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

This concept is better illustrated by an example:

![An example Bayesian network]({{ site.github.url }}/images/bayesian_network_example.png)

Here, \\( A\\) has no parents, \\( B \\)'s parent is \\( A\\), \\( C \\)'s
parent is \\( A\\) and \\( D \\)'s parents are \\( A \\) and \\( B \\). The
probability distribution encoded by this graph is therefore

\\[
    P(A=a, B=b, C=c, D=d) = P(A=a) P(B=b \\mid A=a) P(C=c \\mid A=a)
                            P(D=d \\mid A=a, B=b)
\\]

## Learning bayesian networks, and the inference problem

Bayesian networks are interesting by themselves, but what's even more
interesting is that they can be used to learn something about the distribution
of the random variables they model. Suppose you are given a set of observations
\\( \\mathcal{D} \\), and suppose that the conditional distributions required by
your bayesian network are parametrized by some set of parameters
\\( \\theta \\). The act of learning the distribution which generated the
observations could be described as follows: a parametrization of the model that
approaches the true distribution is a parametrization under which observations
in \\( \\mathcal{D} \\) have a high probability, or alternatively, a high
log-probability (because the logarithm is a monotonically increasing function).
More formally, we are searching \\( \\theta^* \\) such that

\\[
    \\theta^* = \\arg\\max_\\theta \\log P(X_1, \\cdots, X_n)
              = \\arg\\max_\\theta \\sum_{i=1}^n
                \\log P(X_i \\mid \\mathcal{Pa}(X_i))
\\]

This parameter search can be done by various ways, for instance by gradient
descent.

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
