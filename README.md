### pynet ###

A Neural network library primarily to learn differentiable programming (Would not be restricted to just neural networks though) in python inspired from karapathys convnetjs, some nice features from tensorflow and torch/pytorch's autograd structure. (Tried to get the best of all worlds). Please check `main.py` to see how to implement a basic neural network.

This project currently is only implemented in CPU, although I would very much like to implement GPU versions in the future (Just to learn how GPUs work.)

The project is structured as follows:

## Layers

## Networks

## Optimizers

Implemented a Minibatch SGD (which takes batch size and learning rates as hyperparameters and updates parameters).

## Datasets
The idea of datasets like in every other library is to have standard datasets available. Non standard datasets can also be implemented by subclassing `pynet.datasets.Dataset`. In the future, we'd like to add some nice properties to this which would make it easier to work with estimators (dimensions et al)

1. Every dataset should be a subclass of `pynet.datasets.Dataset` and expose `train`, `test` and `valid` splits in the corresponding variables.
2. Implemented a MNISTDataset (http://yann.lecun.com/exdb/mnist/).

## Losses

## Estimators

This is losely based on tensorflow's estimator sturcture. Estimators are capable of working directly with instances of `pynet.datasets.Dataset` (How cool is that!).
Estimators take a network, dataset, optimizer and do the following.
1. Transforms the dataset in a way that networks can consume them. 
    * Good to have - validate the dimensions of the network. 
2. Train the network with the optimizer and dataset and print training and validation (if present) loss after every epoch
3. After training is done, the estimator would save the model (by default and user needs to)
    * We can later add more features to this (like saving the model after every epoch or saving only if a certain parameter crosses a threshold or a previously held threshold)
4. Run the test data on the saved model and give accuracy
5. (Longterm): Create a repository for storing models. (So that they can be later evalutated easily - to promote reproducible research)

This is losely based on the tensorflw

## Regularizers ? (Perhaps later)

## TODO:

1. Fix the dataset - give credit ()
2. Fix the backward propagation in conv layers
3. More layers
4. Implement regularizers
5. Batch Normalization
6. GPU Support, Vectorisation. 
7. How do we think about dimensionality