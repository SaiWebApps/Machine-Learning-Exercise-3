Coursera Machine Learning Exercise #3
===================

Supervised Learning - Multi-Class Problems and Neural Networks

-----------
Prerequisites
-----------
<img src="https://www.gnu.org/software/octave/images//logo.png" width="25" height="25" /> GNU Octave 4.0.0+

<img src="http://itprocurement.unl.edu/software_product_images/matlablogo.jpg" width="100" />

-------------
Files Included In This Exercise
-------------

#### Drivers (Main Files)
* ex3.m - Break multi-class training problem into multiple binary classification problems, and then generate predictions for training set.
* ex3_nn.m - Run neural network to generate predictions for training set.
* submit.m - Submit code to Coursera grader.

#### Datasets
* ex3data1.mat - Training set of handwritten digits.
* ex3weights.mat - Initial weights for neural network.

#### Helper Functions
* displayData.m - Function to help visualize the dataset.
* fmincg.m - Function minimization routine (optimized version of fminunc).
* sigmoid. m - Logistic regression hypothesis function.
* lrCostFunction.m - Regularized logistic regression cost function.

#### One vs. All (Multi-Class to Multiple Binary Classifications)
* oneVsAll.m - For each class c in the original multi-class problem, use logistic regression to isolate "y = c" (1) vs. "y != c" (0).
* predictOneVsAll.m - Predict using oneVsAll classifier function.

#### Neural Networks
* predict.m - Neural network prediction function.

-------------
Essential Concepts
-------------

#### Terminology
* m = number of training examples
* n = number of features
* <img src="https://latex.codecogs.com/gif.latex?\Theta" title="\Theta" /> = hypothesis function weights; parameters
* x = input
* y = actual output
* <img src="https://latex.codecogs.com/gif.latex?h_\Theta(x)" title="h_\Theta(x)" /> = prediction; output of hypothesis function
* <img src="https://latex.codecogs.com/gif.latex?x^{(i)}_j" title="x^{(i)}_j" /> = input for training example i, feature j
* <img src="https://latex.codecogs.com/gif.latex?y^{(i)}" title="y^{(i)}" /> = output for training example i

#### Logistic Regression Hypothesis Function - Sigmoid
<img src="https://latex.codecogs.com/gif.latex?h_\Theta(x)&space;=&space;g(\Theta^Tx)&space;=&space;\frac{1}{1&plus;e^{-\Theta^Tx}}" title="h_\Theta(x) = g(\Theta^Tx) = \frac{1}{1+e^{-\Theta^Tx}}" />

#### Logistic Regression Cost Function (Unregularized)
<img src="https://latex.codecogs.com/gif.latex?J(\Theta)&space;=&space;-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}*log(h_\Theta(x^{(i)}))&space;&plus;&space;(1-y^{(i)})*log(1-h_\Theta(x^{(i)}))" title="J(\Theta) = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}*log(h_\Theta(x^{(i)})) + (1-y^{(i)})*log(1-h_\Theta(x^{(i)}))" />

#### Logistic Regression Cost Function (Regularized)
<img src="https://latex.codecogs.com/gif.latex?J(\Theta)&space;=&space;[-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}*log(h_\Theta(x^{(i)}))&space;&plus;&space;(1-y^{(i)})*log(1-h_\Theta(x^{(i)}))]&space;&plus;&space;[\frac{\lambda}{2m}\sum_{j=1}^{n}\Theta_j^2]" title="J(\Theta) = [-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}*log(h_\Theta(x^{(i)})) + (1-y^{(i)})*log(1-h_\Theta(x^{(i)}))] + [\frac{\lambda}{2m}\sum_{j=1}^{n}\Theta_j^2]" />

#### Gradient Descent (Unregularized)

*For j = [0, n]:*

<img src="https://latex.codecogs.com/gif.latex?\Theta_j&space;:=&space;\Theta_j&space;-&space;\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\Theta(x^{(i)})&space;-&space;y^{(i)})x^{(i)}_j" title="\Theta_j := \Theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\Theta(x^{(i)}) - y^{(i)})x^{(i)}_j" />

#### Gradient Descent (Regularized)

*For j = 0:*

<img src="https://latex.codecogs.com/gif.latex?\Theta_0&space;:=&space;\Theta_0&space;-&space;\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\Theta(x^{(i)})&space;-&space;y^{(i)})x^{(i)}_j" title="\Theta_0 := \Theta_0 - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\Theta(x^{(i)}) - y^{(i)})x^{(i)}_j" />

*For j = [1,n]:*

<img src="https://latex.codecogs.com/gif.latex?\Theta_j&space;:=&space;\Theta_j&space;-&space;\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\Theta(x^{(i)})&space;-&space;y^{(i)})x^{(i)}_j&space;&plus;&space;\frac{\lambda}{m}\Theta_j" title="\Theta_j := \Theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\Theta(x^{(i)}) - y^{(i)})x^{(i)}_j + \frac{\lambda}{m}\Theta_j" />

#### Feedforward Neural Networks

*What is a feedforward neural network?*
* Machine learning model based on the human brain.
* Each successive layer learns from the previous layer (a.k.a, uses it as features). The net effect is that we amalgamate many simple features into a smaller group of more complex features.
* Biggest advantage: Neural networks enable us to create non-linear hypotheses, which could potentially fit the data better than linear ones when dealing with large feature sets.
* Feedforward = signals only move in 1 direction (from input layer to output layer).

*Neural Network Terminology*
* K = number of output classes
* Input layer = input matrix X of size m x (n+1), corresponds to dendrites in neuron
* Hidden layer = intermediate layer between input and output layers, corresponds to cell body in neuron
* Output layer = output vector of size m x K, corresponds to axon and axon terminal in neuron
* <img src="https://latex.codecogs.com/gif.latex?a^{(i)}_u" title="a^{(i)}_u" /> = Activation unit (neuron cell body) u for layer i; note that i = 1 for input layer
* Bias unit = <img src="https://latex.codecogs.com/gif.latex?a^{(i)}_0" title="a^{(i)}_0" /> = 1 for each layer; corresponds to <img src="https://latex.codecogs.com/gif.latex?x_0=1" title="x_0=1" />

*Example of Feedforward Neural Network*

![](http://franck.fleurey.free.fr/NeuralNetwork/images/network.gif)

*Mathematical Representation of Above Neural Network*

Note that <img src="https://latex.codecogs.com/gif.latex?a^{(1)}" title="a^{(1)}" /> is the input layer.

Activation Units in the Hidden Layer

<img src="https://latex.codecogs.com/gif.latex?a^{(2)}_1&space;=&space;g(\theta^{(1)}_{10}&space;&plus;&space;\theta^{(1)}_{11}x_1&space;&plus;&space;\theta^{(1)}_{12}x_2&space;&plus;&space;\theta^{(1)}_{13}x_3&space;&plus;&space;\theta^{(1)}_{14}x_4)&space;=&space;g(\theta^{(1)}_1\cdot&space;x)" title="a^{(2)}_1 = g(\theta^{(1)}_{10} + \theta^{(1)}_{11}x_1 + \theta^{(1)}_{12}x_2 + \theta^{(1)}_{13}x_3 + \theta^{(1)}_{14}x_4) = g(\theta^{(1)}_1\cdot x)" />

<img src="https://latex.codecogs.com/gif.latex?a^{(2)}_2&space;=&space;g(\theta^{(1)}_{20}&space;&plus;&space;\theta^{(1)}_{21}x_1&space;&plus;&space;\theta^{(1)}_{22}x_2&space;&plus;&space;\theta^{(1)}_{23}x_3&space;&plus;&space;\theta^{(1)}_{24}x_4)&space;=&space;g(\theta^{(1)}_2\cdot&space;x)" title="a^{(2)}_2 = g(\theta^{(1)}_{20} + \theta^{(1)}_{21}x_1 + \theta^{(1)}_{22}x_2 + \theta^{(1)}_{23}x_3 + \theta^{(1)}_{24}x_4) = g(\theta^{(1)}_2\cdot x)" />

<img src="https://latex.codecogs.com/gif.latex?a^{(2)}_3&space;=&space;g(\theta^{(1)}_{30}&space;&plus;&space;\theta^{(1)}_{31}x_1&space;&plus;&space;\theta^{(1)}_{32}x_2&space;&plus;&space;\theta^{(1)}_{33}x_3&space;&plus;&space;\theta^{(1)}_{34}x_4)&space;=&space;g(\theta^{(1)}_3\cdot&space;x)" title="a^{(2)}_3 = g(\theta^{(1)}_{30} + \theta^{(1)}_{31}x_1 + \theta^{(1)}_{32}x_2 + \theta^{(1)}_{33}x_3 + \theta^{(1)}_{34}x_4) = g(\theta^{(1)}_3\cdot x)" />

Output Layer

<img src="https://latex.codecogs.com/gif.latex?h^{(1)}_\Theta(x)&space;=&space;a^{(3)}_1&space;=&space;g(\theta^{(2)}_{10}&space;&plus;&space;\theta^{(2)}_{11}a^{(2)}_1&space;&plus;&space;\theta^{(2)}_{12}a^{(2)}_2&space;&plus;&space;\theta^{(2)}_{13}a^{(2)}_3)&space;=&space;g(\theta^{(2)}_1&space;\cdot&space;a^{(2)})" title="h^{(1)}_\Theta(x) = a^{(3)}_1 = g(\theta^{(2)}_{10} + \theta^{(2)}_{11}a^{(2)}_1 + \theta^{(2)}_{12}a^{(2)}_2 + \theta^{(2)}_{13}a^{(2)}_3) = g(\theta^{(2)}_1 \cdot a^{(2)})" />

<img src="https://latex.codecogs.com/gif.latex?h^{(2)}_\Theta(x)&space;=&space;a^{(3)}_2&space;=&space;g(\theta^{(2)}_{20}&space;&plus;&space;\theta^{(2)}_{21}a^{(2)}_1&space;&plus;&space;\theta^{(2)}_{22}a^{(2)}_2&space;&plus;&space;\theta^{(2)}_{23}a^{(2)}_3)&space;=&space;g(\theta^{(2)}_2&space;\cdot&space;a^{(2)})" title="h^{(2)}_\Theta(x) = a^{(3)}_2 = g(\theta^{(2)}_{20} + \theta^{(2)}_{21}a^{(2)}_1 + \theta^{(2)}_{22}a^{(2)}_2 + \theta^{(2)}_{23}a^{(2)}_3) = g(\theta^{(2)}_2 \cdot a^{(2)})" />

-------------
References
-------------
* [Coursera Machine Learning Exercise #3 Instructions](https://github.com/SaiWebApps/Machine-Learning-Exercise-3/blob/master/ex3.pdf)
* GNU Octave Documentation
* Matlab Documentation
