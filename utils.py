import time
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical

class Utils():
    # load train and test dataset
    def load_handwriting_dataset():
        # load dataset
        (trainX, trainY), (testX, testY) = mnist.load_data()
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        return trainX, trainY, testX, testY

    # scale pixels
    def prep_pixels(train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm

    # plot diagnostic learning curves
    def summarize_diagnostics(histories):
        for i in range(len(histories)):
            # plot loss
            pyplot.subplot(2, 1, 1)
            pyplot.title('Cross Entropy Loss')
            pyplot.plot(histories[i].history['loss'], color='blue', label='train')
            pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
            # plot accuracy
            pyplot.subplot(2, 1, 2)
            pyplot.title('Classification Accuracy')
            pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
            pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        pyplot.show()

    # summarize model performance
    def summarize_performance(scores):
        # print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
        # box and whisker plots of results
        pyplot.boxplot(scores)
        pyplot.show()


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
            history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))
            # stores scores
            scores.append(acc)
            histories.append(history)
            print("--- %s seconds ---" % (time.time() - start_time))
        return scores, histories