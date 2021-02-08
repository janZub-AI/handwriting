# baseline cnn model for mnist
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Lambda
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import KFold
from tensorflow.nn import local_response_normalization as LocalResponseNormalization
import time

from utils import Utils

# define cnn model
# will have to keep padding same because of small size of images.
# kernels cannot be oversized either, maybe will use in future for the bigger picture
# not sure how much sense has LocalResponseNormalization for the grey img

def define_model():
	model = Sequential()
	
	model.add(Conv2D(96, (4, 4), strides=(2,2), padding = 'valid',  kernel_initializer='he_uniform', input_shape=(28, 28, 1))) 
	model.add(PReLU())
	# size (28-3)/2 + 1 = 13
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2), strides=(1,1)))
	# size 13 - 1 = 12
	
	## takes in 12 x 12 x 96
	model.add(Conv2D(256, (3, 3), padding = 'valid', kernel_initializer='he_uniform'))
	model.add(PReLU())
	# size 12 - 2 = 10
	model.add(BatchNormalization())
	model.add(MaxPooling2D((3, 3), strides=(1,1)))
	#size 10 - 2 = 8

	## 3 connected layers without pooling or normalizing
	# takes 8 x 8 x 256
	model.add(Conv2D(384, (3, 3), padding = 'valid', kernel_initializer='he_uniform'))
	model.add(PReLU())
	model.add(Conv2D(384, (2, 2), padding = 'valid', kernel_initializer='he_uniform'))
	model.add(PReLU())
	model.add(Conv2D(64, (3, 3), padding = 'valid', kernel_initializer='he_uniform'))
	model.add(PReLU())
	# size 8 - 2 - 1 - 2 = 3

	model.add(Flatten())
	# 3 x 3 x 64 = 576
	model.add(Dense(576, kernel_initializer='he_uniform'))
	model.add(PReLU())
	model.add(Dropout(0.5))
	
	model.add(Dense(576, kernel_initializer='he_uniform'))
	model.add(PReLU())
	model.add(Dropout(0.5))
	
	model.add(Dense(10, activation='softmax'))
	model.add(PReLU())
	# compile model
	opt = SGD(lr=0.01, momentum=0.9, decay=0.01)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		start_time = time.time()
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=256, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
		print("--- %s seconds ---" % (time.time() - start_time))
	return scores, histories


def run_test_harness_with_eval():
	# load dataset
	trainX, trainY, testX, testY = Utils.load_handwriting_dataset()
	# prepare pixel data
	trainX, testX = Utils.prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	Utils.summarize_diagnostics(histories)
	# summarize estimated performance
	Utils.summarize_performance(scores)


run_test_harness_with_eval()