from __future__ import absolute_import
from __future__ import print_function
import cPickle as pickle
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.regularizers import Consensus
from keras import regularizers
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras import callbacks

import sys

'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''

class History(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in self.totals.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v / self.seen)

        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def on_train_end(self, logs={}):
        with open('history.pickle', 'w') as f:
            pickle.dump(self.history, f)

batch_size = 32
nb_classes = 10
nb_epoch = 100
data_augmentation = False

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 32, 32
# number of convolutional filters to use at each layer
nb_filters = [32, 64]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]
# the CIFAR10 images are RGB
image_dimensions = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if sys.argv[1] == '1':
    X_train=X_train[0:25000]
    y_train=y_train[0:25000]
if sys.argv[1] == '0':
    X_train=X_train[25000:]
    y_train=y_train[25000:]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

init_method = 'he_normal'
model.add(Convolution2D(nb_filters[0], image_dimensions, nb_conv[0], nb_conv[0], W_regularizer=Consensus(rho=regularizers.RHO), b_regularizer=Consensus(rho=regularizers.RHO), init=init_method, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[0], nb_filters[0], nb_conv[0], nb_conv[0], W_regularizer=Consensus(rho=regularizers.RHO), b_regularizer=Consensus(rho=regularizers.RHO), init=init_method))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[0], nb_pool[0])))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters[1], nb_filters[0], nb_conv[0], nb_conv[0], border_mode='full', W_regularizer=Consensus(rho=regularizers.RHO), b_regularizer=Consensus(rho=regularizers.RHO), init=init_method))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[1], nb_filters[1], nb_conv[1], nb_conv[1], W_regularizer=Consensus(rho=regularizers.RHO), b_regularizer=Consensus(rho=regularizers.RHO), init=init_method))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[1], nb_pool[1])))
model.add(Dropout(0.25))

model.add(Flatten())
# the image dimensions are the original dimensions divided by any pooling
# each pixel has a number of filters, determined by the last Convolution2D
# layer
model.add(Dense(nb_filters[-1] * (shapex / nb_pool[0] / nb_pool[1]) *
                (shapey / nb_pool[0] / nb_pool[1]), 512, W_regularizer=Consensus(rho=regularizers.RHO), b_regularizer=Consensus(rho=regularizers.RHO), init=init_method))
model.add(Activation('relu'))
model.add(Dropout(0.50))

model.add(Dense(512, nb_classes,W_regularizer=Consensus(rho=regularizers.RHO), b_regularizer=Consensus(rho=regularizers.RHO), init=init_method))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.load_weights('weights.hd5')
#model.save_weights('weights.hd5')
if len(sys.argv) > 1:
    port = 6000 + int(sys.argv[1])
    server_address = 'tcp://bordeaux.local.:%d' % port
    print('using weight server @ %s' % server_address)
    weight_sync = callbacks.WeightSynchronizer(server_address, frequency=512)


print("Not using data augmentation or normalization")

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
history = History()
model.fit(X_train, Y_train,
          batch_size=batch_size,
          callbacks=[weight_sync, history],
          verbose=2,
          nb_epoch=nb_epoch,show_accuracy=True)
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)
