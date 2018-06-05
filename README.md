# A (Yet Another) PyTorch (0.3.0) Implementation of Residual Networks

This is a [PyTorch](http://pytorch.org/) implementation of the
Residual Network architecture with basic blocks as described
paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun.
This implementation gets a CIFAR10 test accuracy of %92-93 percent
when 18 layers are used (in the paper they also achieve %93 percent with 18 layers) 
Different annealing schemes or changing the order of the batchnorm and relu does not 
seem to make any significant improvements but this requires more testing to make sure. 
SGD with a decreasing learning rate and weight decay=0.0005 seems to perform the
best among all optimizers. Applying random rotations, flips and crops to the training set
at the beginning of each cycle to boost the variety in training set seems to improve
the accuracy by about %1-2 percent. ZCA data whitening and layer-wise properties 
(setting the weight decays for bias layers to 0) are also implemented.

# Architectural details

The network contains an initial convolution + 3 stages of residual blocks. 
The number of residual blocks inside each stage as well as the initial width 
of the first block can be given as inputs to the network constructer. The width 
then increases as 3 -> width -> 2*width -> 4*width -> 8*width. Convolution 
shortcut connections are used as the "identity" mappings. 

# Implementation details

The code has been implemented for pytorch 0.3. So if you want to run it in 
pytorch 0.4 you need to change the way some scalars are handled (in particular
change .sum() to .sum().item() and .data[0] to .item()). 


# Results for various settings

### Implementation of ResNet18 with exponentially decreasing learning rate

![](images/Graph2.png)


Best test accuracy: 0.92490, training accuracy: 0.99632 (cost:0.02136)

|  Class  |  Score                                      |
| ------- | ------------------------------------------- |
|  plane  | Precision:0.92, Recall: 0.93, F1 norm: 0.92 |
|  car    | Precision:0.95, Recall: 0.96, F1 norm: 0.96 |
|  bird   | Precision:0.92, Recall: 0.90, F1 norm: 0.91 | 
|  cat    | Precision:0.83, Recall: 0.85, F1 norm: 0.84 |
|  deer   | Precision:0.93, Recall: 0.94, F1 norm: 0.93 |
|  dog    | Precision:0.89, Recall: 0.89, F1 norm: 0.89 |
|  frog   | Precision:0.95, Recall: 0.95, F1 norm: 0.95 |
|  horse  | Precision:0.96, Recall: 0.93, F1 norm: 0.95 |
|  ship   | Precision:0.95, Recall: 0.95, F1 norm: 0.95 |
|  truck  | Precision:0.95, Recall: 0.95, F1 norm: 0.95 |



### Implementation of ResNet with 62 layers with exponentially decreasing learning rate

![](images/Graph3.png)

Best test accuracy: 0.91590, training accuracy: 0.99350 (cost:0.01992)

|  Class  |  Score                                      |
| ------- | ------------------------------------------- |
|  plane  | Precision:0.91, Recall: 0.93, F1 norm: 0.92 |
|  car    | Precision:0.95, Recall: 0.95, F1 norm: 0.95 |
|  bird   | Precision:0.89, Recall: 0.88, F1 norm: 0.89 | 
|  cat    | Precision:0.82, Recall: 0.82, F1 norm: 0.82 |
|  deer   | Precision:0.91, Recall: 0.93, F1 norm: 0.92 |
|  dog    | Precision:0.86, Recall: 0.86, F1 norm: 0.86 |
|  frog   | Precision:0.95, Recall: 0.93, F1 norm: 0.94 |
|  horse  | Precision:0.94, Recall: 0.93, F1 norm: 0.94 |
|  ship   | Precision:0.96, Recall: 0.94, F1 norm: 0.95 |
|  truck  | Precision:0.93, Recall: 0.94, F1 norm: 0.94 |


### Discussion
Note that a high precision, low recall for an object X means that the network is very cautious 
about this object. So for many objects which are X, the network might say "this is not X", but if it says
it is "X" then it is very likely to be correct. So it is true guesses are very precise but can not 
recall most X.

A high recall but low precision for an object X means that the network is very good at finding all the objects
which are X but it overshoots. So it identifies most of all the objects X as X but it also identifies many other
which are not X as X as well. So it is not very precise but recalls all the X.

In this case we see that cats and dogs have both lower recall and precision where as others have higher for both.
This likely means that the program confuses cats and dogs with each other. A full confusion matrix
can sort this out (to be implemented in future).

# Requirements

Pytorch 0.3 and internet connection to download the datasets.

# Author
Sina Tureli
Yet Another Implementation of Residual Networks


# Licensing

This repository is
[Apache-licensed](https://github.com/bamos/densenet.pytorch/blob/master/LICENSE).
