# baseline cnn model for mnist
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import load_model
from utils import Utils

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
	
# run the test harness for evaluating a model
def run_test_harness_with_eval():
	# load dataset
	trainX, trainY, testX, testY = Utils.load_handwriting_dataset()
	# prepare pixel data
	trainX, testX = Utils.prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = Utils.evaluate_model(trainX, trainY)
	# learning curves
	Utils.summarize_diagnostics(histories)
	# summarize estimated performance
	Utils.summarize_performance(scores)

def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = Utils.load_handwriting_dataset()
	# prepare pixel data
	trainX, testX = Utils.prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	# save model
	model.save('final_model.h5')

def eval():
	# load dataset
	trainX, trainY, testX, testY = Utils.load_handwriting_dataset()
	# prepare pixel data
	trainX, testX = Utils.prep_pixels(trainX, testX)
	# load model
	model = load_model('final_model.h5')
	# evaluate model on test dataset
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))


# entry point, run the test harness
eval()