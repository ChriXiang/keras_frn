# Filter Response Normalization Layer in Keras

This repository replicates [this repository](https://github.com/gupta-abhay/pytorch-frn) in keras.

paper: [Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737)

## Features

see https://github.com/gupta-abhay/pytorch-frn

## Example Usage

```
>> ...
>> x = Conv2D(64, (3,3))(x)
>> x = FilterResponseNorm2d(64)(x) #specify input channel numbers
>> x = Activation('relu')(x)
>> ...
```
