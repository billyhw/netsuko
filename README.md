# netzuko

<img src="/images/nezuko_2.jpg" width="300" height="400">

# Valentine's Update

The package now supports more features, including Xavier initialization, ReLU activation, Stochastic Gradient Descent, Dropout, and ADAM optimization. Continuous outputs and multiple logistic outputs are also now supported.

Batch-normalization has been implemented. Due to its complexity it is currently stored in an alternative branch "batchnorm" to simplify continuing development of the package.

# Overview

This R package contains a simple implementation of a "textbook" multi-layer neural network.

The purpose of creating this package is to demonstrate to interested viewers how neural networks are "built from scratch". 
This may be of some interests to viewers who like to learn neural networks, or users of neural networks who are potentially bewildered
by the many intricacies of modern neural networks. 

At present the functionalities of the package are limited, but will however be continuously updated. This is the whole point of creating this package after all: to demonstrate how earlier neural networks evolve to modern neural networks through various scientific advancements.

# Installation
```
install.packages("devtools")
devtools::install_github("billyhw/netzuko")
```
