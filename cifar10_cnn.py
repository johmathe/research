from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.regularizers import Consensus
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

batch_size = 32
nb_classes = 10
nb_epoch = 200
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
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

w_reg = Consensus()
b_reg = Consensus()
model = Sequential()

model.add(Convolution2D(nb_filters[0], image_dimensions, nb_conv[0], nb_conv[0], W_regularizer=w_reg, b_regularizer=b_reg, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[0], nb_filters[0], nb_conv[0], nb_conv[0]))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[0], nb_pool[0])))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters[1], nb_filters[0], nb_conv[0], nb_conv[0], border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[1], nb_filters[1], nb_conv[1], nb_conv[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[1], nb_pool[1])))
model.add(Dropout(0.25))

model.add(Flatten())
# the image dimensions are the original dimensions divided by any pooling
# each pixel has a number of filters, determined by the last Convolution2D
# layer
model.add(Dense(nb_filters[-1] * (shapex / nb_pool[0] / nb_pool[1]) *
                (shapey / nb_pool[0] / nb_pool[1]), 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

if len(sys.argv) > 1:
    port = 6000 + int(sys.argv[1])
    server_address = 'tcp://bordeaux.local.:%d' % port
    print('using weight server @ %s' % server_address)
    weight_sync = callbacks.WeightSynchronizer(server_address, frequency=128)

print("Not using data augmentation or normalization")

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

if len(sys.argv) > 1:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              callbacks=[weight_sync],
              shuffle=True,
              nb_epoch=nb_epoch,validation_split=0.1,show_accuracy=True)
else:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              nb_epoch=nb_epoch)
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)
