import numpy as np
from cnn_md1_funs import *


## Returns gradient for all the paramaters in each iteration
def ConvNet(image, label, filt1, filt2, bias1, bias2, W_fc1, bias_fc1, fcnum, theta3, bias3):
	## Calculating first Convolution layer
		
	## l - channel
	## w - size of square image
	## l1 - No. of filters in Conv1
	## l2 - No. of filters in Conv2
	## w1 - size of image after conv1
	## w2 - size of image after conv2

	#convnet order
	#image_mnist(1,28,28)->filt1(10,10,32,s=1)->max_pool(2,2)->filt2(5,5,16,s=1)->max_pool(2,2)->fc1(1024)->classify(10)->softmax+cross_entropy

	(l, w, w) = image.shape	# (1, 28,28) for mnist
	l1 = len(filt1) #32 filters
	l2 = len(filt2) #16 filters
	( _, f1, f1) = filt1[0].shape #10 by 10
	(_, f2, f2) = filt2[0].shape  # 5 by 5
	fp=2 #pool size
	s=1 #stride 1 for all filers pools
	w1 = int(np.floor((w-f1+s)/s)) #size of image after filter1, check padding

	conv1=conv2d(image, w, filt1, bias1, l1, f1, s)

	#apply first pooling after filter1
	pooled_layer1 = maxpool(conv1, fp, s)

	#w2 is size of pooled_layer1
	w2_int=pooled_layer1.shape[1]
	w2=int(np.floor(w2_int-f2+s)/s)

	conv2=conv2d(pooled_layer1, w2_int, filt2, bias2, l2, f2, s)

	## Pooled layer with 2*2 size and stride 1
	pooled_layer2 = maxpool(conv2, fp, s)

	#flatten the pooled layer output
	flatten1 = pooled_layer2.reshape((-1,1))

	#1024 neuron fully connected layer
	#W is weight metrics with size (fcnum,flatten.shape[0])
	fc1=W_fc1.dot(flatten1)+bias_fc1
	fc1[fc1<=0]=0 #relu

	#classify layer (10,1)
	out = theta3.dot(fc1) + bias3

	#find loss
	loss, probs = softmax_entropy(out, label)

	#define accuracy
	if np.argmax(out)==np.argmax(label):
		acc=1
	else:
		acc=0

	################ DO BACKPROPAGATION ##################################

	dout = probs - label	#	dL/dout
	
	dtheta3 = dout.dot(fc1.T) 		##	dL/dtheta3

	dbias3 = sum(dout.T).T.reshape((10,1))		##	dbias3

	dfc1 = theta3.T.dot(dout)		##	dL/dfc1

	dW_fc1=dfc1.dot(flatten1.T) # dL/dW_fc1 fcnum fully connected layer

	dbias_fc1=sum(dfc1.T).T.reshape((fcnum,1)) #dL/dbias_fc1

	dflatten1=W_fc1.T.dot(dfc1) #dL/dflatten1

	#second layer filter2->maxpool
	dpooled_layer2 = dflatten1.T.reshape(pooled_layer2.shape)

	dconv2 = np.zeros((l2, w2, w2))

	backprop_dconv(dpooled_layer2, dconv2, conv2, l2, w2, fp, s)

	dpooled_layer1=np.zeros(pooled_layer1.shape)
	dfilt2 = {}
	dbias2 = {}
	for c in range(0,l2):
		dfilt2[c] = np.zeros((l1,f2,f2))
		dbias2[c] = 0


	#filter2 is between pooled_layer1 and conv2
	for c in range(0,l2):
		for i in range(0,w2,s):
			for j in range(0,w2,s):
				dfilt2[c]+=dconv2[c,i,j]*pooled_layer1[:,i:i+f2,j:j+f2]
				dpooled_layer1[:,i:i+f2,j:j+f2]+=dconv2[c,i,j]*filt2[c]
		dbias2[c] = np.sum(dconv2[c])


	#first filter1->maxpool layer
	dconv1 = np.zeros((l1, w1, w1))
	dfilt1 = {}
	dbias1 = {}
	for c in range(0,l1):
		dfilt1[c] = np.zeros((l,f1,f1))
		dbias1[c] = 0

	#get conv1 by using pooling output
	backprop_dconv(dpooled_layer1, dconv1, conv1, l1, w1, fp, s)

	for c in range(0,l1):
		for i in range(0,w1,s):
			for j in range(0,w1,s):
				dfilt1[c]+=dconv1[c,i,j]*image[:,i:i+f1,j:j+f1]

		dbias1[c] = np.sum(dconv1[c])

	if 0:
		print('BEGIN SHOWING SHAPES')
		print('filt1[0]:', filt1[0].shape)
		print('bias1:', len(bias1))
		print('conv1:',conv1.shape)
		print('pooled_layer1:', pooled_layer1.shape)
		print('filt2[0]:', filt2[0].shape)
		print('bias2:', len(bias2))
		print('conv2:',conv2.shape)
		print('pooled_layer2:', pooled_layer2.shape)
		print("flatten1:",flatten1.shape)
		print('W_fc1:', W_fc1.shape)
		print ('bias_fc1:', bias_fc1.shape)
		print('fc1:', fc1.shape)
		print('theta3:', theta3.shape)
		print ('bias3:', bias3.shape)
		print('out:', out.shape)
		print('BEGIN SHOWING BACKPROP SHAPES')
		print('dout:', dout.shape)
		print('dtheta3:', dtheta3.shape)
		print('dbias3:',dbias3.shape)
		print('dfc1:', dfc1.shape)
		print('dW_fc1:', dW_fc1.shape)
		print('dbias_fc1:', dbias_fc1.shape)
		print('dflatten1:', dflatten1.shape)
		print('dpooled_layer2:', dpooled_layer2.shape)
		print('dconv2', dconv2.shape)
		print('dfilt2[0]:', dfilt2[0].shape)
		print('dbias2:', len(dbias2))
		print('dpooled_layer1:', dpooled_layer1.shape)
		print('dconv1:', dconv1.shape)
		print('dfilt1[0]:', dfilt1[0].shape)
		print('dbias1:', len(dbias1))
		print('END SHOWING SHAPES')
	
	return [dfilt1, dfilt2, dbias1, dbias2, dW_fc1, dbias_fc1, dtheta3, dbias3, loss, acc]


## Predict class of each row of matrix X
def predict(yin, image, filt1, filt2, bias1, bias2, W_fc1, bias_fc1, theta3, bias3):
	
	## l - channel
	## w - size of square image
	## l1 - No. of filters in Conv1
	## l2 - No. of filters in Conv2
	## w1 - size of image after conv1
	## w2 - size of image after conv2

	(l, w, w) = image.shape  # (1, 28,28) for mnist
	l1 = len(filt1)  # 32 filters
	l2 = len(filt2)  # 16 filters
	(_, f1, f1) = filt1[0].shape  # 10 by 10
	(_, f2, f2) = filt2[0].shape  # 5 by 5
	fp = 2  # pool size
	s = 1  # stride 1 for all filers pools

	conv1 = conv2d(image, w, filt1, bias1, l1, f1, s)

	# apply first pooling after filter1
	pooled_layer1 = maxpool(conv1, fp, s)

	# w2 is size of pooled_layer1
	w2_int = pooled_layer1.shape[1]

	conv2 = conv2d(pooled_layer1, w2_int, filt2, bias2, l2, f2, s)

	## Pooled layer with 2*2 size and stride 1
	pooled_layer2 = maxpool(conv2, fp, s)

	# flatten the pooled layer output
	flatten1 = pooled_layer2.reshape((-1, 1))

	# 1024 neuron fully connected layer
	fc1 = W_fc1.dot(flatten1) + bias_fc1
	fc1[fc1 <= 0] = 0  # relu

	out = theta3.dot(fc1) + bias3  # 10*1

	#find losses and probs
	label = np.zeros((out.shape[0], 1))
	label[int(yin), 0] = 1
	loss, probs = softmax_entropy(out, label)

	return np.argmax(probs), np.max(probs), loss

## Returns all the trained parameters
def AdaGradDescent(batch, LEARNING_RATE, w, l, filt1, filt2, bias1, bias2, W_fc1, bias_fc1, fcnum, theta3,
				   bias3, loss, acc, gdfilt1, gdfilt2, gdbias1, gdbias2, gdW_fc1, gdbias_fc1, gdtheta3, gdbias3):
	"""
	Calculates the adagrad from each run
	:param batch: batch images
	:param LEARNING_RATE: learning rate
	:param w: image size
	:param l: image channel
	:param filt1: filter1 matrix
	:param filt2: filter2 matrix
	:param bias1: filter1 bias
	:param bias2: filter2 bias
	:param W_fc1: fully connected weights
	:param bias_fc1: fully connected bias
	:param fcnum: fully connected neuron number
	:param theta3: output layer weights
	:param bias3: output layer biases
	:param loss: cross entropy loss
	:param acc: accuracy
	:param gdfilt1: ada global grad sum
	:param gdfilt2: ada global grad sum
	:param gdbias1: ada global grad sum
	:param gdbias2: ada global grad sum
	:param gdW_fc1: ada global grad sum
	:param gdbias_fc1: ada global grad sum
	:param gdtheta3: ada global grad sum
	:param gdbias3: ada global grad sum
	:return:
	"""
	# reshape batch image for use
	X = batch[:, 0:-1]
	X = X.reshape(len(batch), l, w, w)
	y = batch[:, -1]
	fudge_factor = 1e-6

	n_correct = 0
	loss_ = 0
	batch_size = len(batch)
	dfilt2 = {}
	dfilt1 = {}
	dbias2 = {}
	dbias1 = {}

	for k in range(0, len(filt2)):
		dfilt2[k] = np.zeros(filt2[0].shape)
		dbias2[k] = 0

	for k in range(0, len(filt1)):
		dfilt1[k] = np.zeros(filt1[0].shape)
		dbias1[k] = 0

	dtheta3 = np.zeros(theta3.shape)
	dbias3 = np.zeros(bias3.shape)

	dW_fc1 = np.zeros(W_fc1.shape)
	dbias_fc1 = np.zeros(bias_fc1.shape)

	for i in range(0, batch_size):
		image = X[i]
		label = np.zeros((theta3.shape[0], 1))
		label[int(y[i]), 0] = 1
		## Fetching gradient for the current parameters
		[dfilt1_, dfilt2_, dbias1_, dbias2_, dW_fc1_, dbias_fc1_, dtheta3_, dbias3_, loss_im, acc_] = ConvNet(image,
																											  label,
																											  filt1,
																											  filt2,
																											  bias1,
																											  bias2,
																											  W_fc1,
																											  bias_fc1,
																											  fcnum,
																											  theta3,
																											  bias3)

		for j in range(0, len(filt2)):
			dfilt2[j] += dfilt2_[j]
			dbias2[j] += dbias2_[j]

		for j in range(0, len(filt1)):
			dfilt1[j] += dfilt1_[j]
			dbias1[j] += dbias1_[j]

		dtheta3 += dtheta3_
		dbias3 += dbias3_
		dW_fc1 += dW_fc1_
		dbias_fc1 += dbias_fc1_

		loss_ += loss_im
		n_correct += acc_

	# add to sums
	# sum of all gradient updates
	for j in range(0, len(filt2)):
		gdfilt2[j] += (dfilt2[j] / batch_size) ** 2
		gdbias2[j] += (dbias2[j] / batch_size) ** 2

	for j in range(0, len(filt1)):
		gdfilt1[j] += (dfilt1[j] / batch_size) ** 2
		gdbias1[j] += (dbias1[j] / batch_size) ** 2

	gdtheta3 += (dtheta3 / batch_size) ** 2
	gdbias3 += (dbias3 / batch_size) ** 2
	gdW_fc1 += (dW_fc1 / batch_size) ** 2
	gdbias_fc1 += (dbias_fc1 / batch_size) ** 2

	for j in range(0, len(filt1)):
		filt1[j] -= (dfilt1[j] / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdfilt1[j])
		bias1[j] -= (dbias1[j] / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdbias1[j])
	for j in range(0, len(filt2)):
		filt2[j] -= (dfilt2[j] / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdfilt2[j])
		bias2[j] -= (dbias2[j] / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdbias2[j])

	theta3 -= (dtheta3 / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdtheta3)
	bias3 -= (dbias3 / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdbias3)

	W_fc1 -= (dW_fc1 / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdW_fc1)
	bias_fc1 -= (dbias_fc1 / batch_size) * LEARNING_RATE / np.sqrt(fudge_factor + gdbias_fc1)

	loss_ = loss_ / batch_size
	loss.append(loss_)
	accuracy = float(n_correct) / batch_size
	acc.append(accuracy)

	return [filt1, filt2, bias1, bias2, W_fc1, bias_fc1, theta3, bias3, loss, acc, gdfilt1, gdfilt2, gdbias1,
			gdbias2, gdW_fc1, gdbias_fc1, gdtheta3, gdbias3]
