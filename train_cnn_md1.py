import numpy as np
import pickle
import time
import sys

from convnet_md1 import *
from remtime import *
from cnn_md1_funs import *

## Hyperparameters
NUM_OUTPUT = 10
LEARNING_RATE =0.01 # 0.01	#learning rate
IMG_WIDTH = 28
IMG_DEPTH = 1
FILTER_SIZE1=10
FILTER_SIZE2=5
STRIDE=1
POOLSIZE=2
NUM_FILT1 = 32
NUM_FILT2 = 16
FCNUM=1024 #fully connected layer neuron number
BATCH_SIZE = 100
NUM_EPOCHS = 5
np.random.seed(123)

PICKLE_FILE = 'output_cnn_md1.pickle'

## Data extracting
m =10000
X = extract_data('t10k-images-idx3-ubyte.gz', m, IMG_WIDTH)
y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))
test_data = np.hstack((X,y_dash))


m =20000
X = extract_data('train-images-idx3-ubyte.gz', m, IMG_WIDTH)
y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
print(np.mean(X), np.std(X))
X-= int(np.mean(X))
X/= int(np.std(X))
train_data = np.hstack((X,y_dash))
# print(X.shape, train_data.shape)

np.random.shuffle(train_data)


NUM_IMAGES = train_data.shape[0]

## Initializing the parameters
filt1 = {}
filt2 = {}
bias1 = {}
bias2 = {}

#initalize global sum of grads for adagrad
gdfilt1={}
gdfilt2={}
gdbias1={}
gdbias2={}


for i in range(0,NUM_FILT1):
	filt1[i]=initialize_weight_tnorm((IMG_DEPTH,FILTER_SIZE1,FILTER_SIZE1))
	bias1[i] = 0.1
	gdfilt1[i]=np.zeros(filt1[i].shape)
	gdbias1[i]=0
	# v1[i] = 0
for i in range(0,NUM_FILT2):
	filt2[i] = initialize_weight_tnorm((NUM_FILT1, FILTER_SIZE2, FILTER_SIZE2))
	bias2[i] = 0.1
	gdfilt2[i] = np.zeros(filt2[i].shape)
	gdbias2[i] = 0
	# v2[i] = 0
w1 = int(np.floor((IMG_WIDTH-FILTER_SIZE1+STRIDE)/STRIDE))
w1_pool1=int(np.floor((w1-POOLSIZE+STRIDE)/STRIDE)) #check
w2 = int(np.floor((w1_pool1-FILTER_SIZE2+STRIDE)/STRIDE))
w2_pool2=int(np.floor((w2-POOLSIZE+STRIDE)/STRIDE))

W_fc1=initialize_weight_tnorm((FCNUM, int(w2_pool2*w2_pool2*NUM_FILT2)))

theta3=initialize_weight_tnorm((NUM_OUTPUT, FCNUM))


bias3 = 0.1*np.ones((NUM_OUTPUT,1))
bias_fc1=0.1*np.ones((FCNUM,1))

gdW_fc1=np.zeros(W_fc1.shape)
gdbias_fc1=np.zeros(bias_fc1.shape)
gdtheta3=np.zeros(theta3.shape)
gdbias3=np.zeros(bias3.shape)

bloss = []
acc = []

#Define if this is a train run or prediction
ISTRAIN=int(sys.argv[1])

if ISTRAIN==1:
	f = open("train_cnn_md1_log.txt", "w+")
	print("Learning Rate:"+str(LEARNING_RATE)+", Batch Size:"+str(BATCH_SIZE))
	f.write("Learning Rate:"+str(LEARNING_RATE)+", Batch Size:"+str(BATCH_SIZE)+"\n")

	## Training start here
	for epoch in range(0,NUM_EPOCHS):
		np.random.shuffle(train_data)
		batches = [train_data[k:k + BATCH_SIZE] for k in range(0, NUM_IMAGES, BATCH_SIZE)]
		x=0
		for batch in batches:
			stime = time.time()
			out = AdaGradDescent(batch, LEARNING_RATE, IMG_WIDTH, IMG_DEPTH, filt1, filt2, bias1, bias2, W_fc1, bias_fc1, FCNUM, theta3, bias3, bloss, acc,
									  gdfilt1, gdfilt2, gdbias1, gdbias2, gdW_fc1, gdbias_fc1, gdtheta3, gdbias3)
			[filt1, filt2, bias1, bias2, W_fc1, bias_fc1, theta3, bias3, bloss, acc,  gdfilt1, gdfilt2, gdbias1, gdbias2, gdW_fc1, gdbias_fc1, gdtheta3, gdbias3] = out
			epoch_acc = round(np.sum(acc[int(np.floor(epoch*NUM_IMAGES/BATCH_SIZE)):])/(x+1),2)

			per = float(x+1)/len(batches)*100
			print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(NUM_EPOCHS)+", Loss:"+str(bloss[-1])+", Batch Acc:"+str(acc[-1]*100)+"%, Epoch Acc:"+str(epoch_acc*100)+"%")
			f.write("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(NUM_EPOCHS)+", Loss:"+str(bloss[-1])+", Batch Acc:"+str(acc[-1]*100)+"%, Epoch Acc:"+str(epoch_acc*100)+"%\n")

			ftime = time.time()
			deltime = ftime-stime
			remtime = (len(batches)-x-1)*deltime+deltime*len(batches)*(NUM_EPOCHS-epoch-1)
			print(printTime(remtime))
			f.write(printTime(remtime)+"\n")
			x+=1


	## saving the trained model parameters
	with open(PICKLE_FILE, 'wb') as file:
		pickle.dump(out, file)
	f.close()

else:
	f=open("predict_cnn_md1_log.txt", 'w+')
	print("predicting for test dataset with trained model file:"+str(PICKLE_FILE))
	f.write("predicting for test dataset with trained model file:"+str(PICKLE_FILE)+"\n")

	## Opening the saved model parameter
	pickle_in = open(PICKLE_FILE, 'rb')
	out = pickle.load(pickle_in)

	[filt1, filt2, bias1, bias2, W_fc1, bias_fc1, theta3, bias3, bloss, acc,  gdfilt1, gdfilt2, gdbias1, gdbias2, gdW_fc1, gdbias_fc1, gdtheta3, gdbias3] = out

	## Computing Test accuracy
	X = test_data[:,0:-1]
	X = X.reshape(len(test_data), IMG_DEPTH, IMG_WIDTH, IMG_WIDTH)
	y = test_data[:,-1]
	corr = 0
	loss=0
	print("Computing accuracy over test set:")
	f.write("Computing accuracy over test set:\n")
	test_img_cnt=int(len(test_data)/1)
	for i in range(0,test_img_cnt):
		image = X[i]
		digit, prob, bloss = predict(y[i], image, filt1, filt2, bias1, bias2, W_fc1, bias_fc1, theta3, bias3)
		loss+=bloss

		if digit==y[i]:
			corr+=1
		if (i+1)%int(0.05*test_img_cnt)==0:
			print(str(float(i+1)/test_img_cnt*100)+"% Completed")
			f.write(str(float(i+1)/test_img_cnt*100)+"% Completed\n")
			test_acc = float(corr)/(i+1)*100
			print("Test Set Accuracy:",str(test_acc)+"% Test Set Loss:", str(loss[-1]/(i+1)))
			f.write("Test Set Accuracy: "+str(test_acc)+"% Test Set Loss: "+str(loss[-1]/(i+1))+"%\n")
	test_acc = float(corr) / test_img_cnt * 100
	print("Final Test Set Accuracy:", str(test_acc)+"% Test Set Loss:", str(loss[-1]/test_img_cnt))
	f.write("Final Test Set Accuracy: "+str(test_acc)+"% Test Set Loss: "+str(loss[-1]/test_img_cnt)+"\n")
	f.close()
