# CIFAR-10 CNN Tutorial Extension

This project is a first-pass PyTorch image classification workflow using the CIFAR-10 dataset. It starts from the official PyTorch CIFAR-10 tutorial model, then adds a few pieces that make the workflow closer to what I would use in a real supervised learning project: a validation split, train-only normalization statistics, early stopping, learning curves, and test-set diagnostics.

The goal here is not to build a state-of-the-art CIFAR-10 model. The goal is to show the fundamentals of training, validating, and evaluating a convolutional neural network in PyTorch without hiding the workflow behind a high-level training framework.

## What This Project Shows

- Loading CIFAR-10 with `torchvision`
- Computing normalization statistics from the training split only
- Creating reproducible train, validation, and test data loaders
- Training a small convolutional neural network with PyTorch
- Tracking train and validation loss/accuracy over epochs
- Applying early stopping based on validation loss
- Restoring the best validation-loss model before test evaluation
- Evaluating the final model with scikit-learn metrics
- Saving diagnostic plots and metric tables

## Methodology

The original CIFAR-10 training set is split into a training subset and validation subset. Normalization mean and standard deviation are computed only from the training subset, then applied consistently to the training, validation, and test sets.

The model is trained on the training subset while validation loss is monitored after each epoch. If validation loss stops improving for several epochs, training stops early and the best model weights are restored.

The official CIFAR-10 test set is used only once after training is complete.

## Outputs

The notebook generates the following artifacts:

- [Training and validation curves](figures/training_validation_curves.png)
- [Confusion matrix](figures/confusion_matrix.png)
- [ROC-AUC curves](figures/roc_auc_curves.png)
- [Classification metrics table](tables/test_classification_metrics.md)
- [Classification metrics CSV](tables/test_classification_metrics.csv)

These outputs make it easier to understand how the model is behaving instead of relying on one accuracy number.

## Current Model

The CNN architecture is intentionally close to the PyTorch tutorial network:

- two convolution layers
- max pooling
- three fully connected layers
- cross-entropy loss
- SGD with momentum

Because the model is intentionally simple, performance is expected to be modest compared with modern CIFAR-10 models. This baseline is still useful because it makes the full workflow easy to understand before adding more advanced architecture and training improvements.

## Next Steps

Natural next improvements would be:

- data augmentation, such as random crops and horizontal flips
- batch normalization
- weight decay
- a learning-rate scheduler
- a stronger CNN or small ResNet-style architecture
- misclassified-image inspection
- per-class accuracy plots

These changes would likely improve performance while keeping the same train, validation, and test workflow used here.
