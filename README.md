## A Numerically Stable Deep Convolutional Neural Network Implementation from Scratch Using Numpy

CONVNET MODEL 1:

image (1 channel, 28x28)
     |
     |
filter 1 (32 channels,10x10)
	 |
	 |
max-pool (2x2)
	 |
	 | 
filter 2 (16 channels, 5x5)
	 |
	 | 
max-pool(2x2)
	 |
	 | 
fully connected (1024 neurons)
	 |
	 | 
output (10 classes/neurons)
	 |
	 | 
softmax+cross-entropy with Adagrad optimizer

- Stride is 1 for all filters and max-pool.
- Padding is VALID.
- Activation used is ReLU.
- Weights initialized with truncated normal distribution.
- Biases initialized to a small constant value.
- Numerically stable softmax and cross-entropy definitions are implemented to train deeper models for multiple epochs.
- Adagrad optimizer is implemented to adaptively adjust learning rate for each parameter.
- Forward and backward propagation for filters and pool layers are implemented using numpy only.


### How to Train the Model
To train the model using MNIST data set, run as:
```
python train_cnn_md1.py 1
```
Training generates output_cnn_md1.pickle file that stores trained model variables for prediction and a log file train_cnn_md1_log.txt showing loss and accuracy progress. These files are  generated from a prior run and available in the repo.

Training uses 5 epochs with 100 images/batch and total of 20000 images per epoch. (less than half of total MNIST training images.)

### How to Predict the Model
To predict the model using trained variables in pickle file over MNIST test dataset, run:
```
python train_cnn_md1.py 0
```
The test accuracy and loss progress will be logged in predict_cnn_md1_log.txt. This file is also available in the repo.

The trained model predicts the MNIST test data with ~94% accuracy.


### Acknowledgements:
MNIST implemnetation from below repo was used as a starting point for implementing this deeper CNN model and to prepare MNIST data: 
https://github.com/zishansami102/CNN-from-Scratch/.

Another repo with useful ideas and insights:
https://github.com/dorajam/Convolutional-Network.



