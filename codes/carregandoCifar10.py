#!/usr/bin/python


import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

num_classes = 10
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
print input_shape

train_file1 = './cifar-10-batches-py/data_batch_1'
train_file2 = './cifar-10-batches-py/data_batch_2'
train_file3 = './cifar-10-batches-py/data_batch_3'
train_file4 = './cifar-10-batches-py/data_batch_4'
train_file5 = './cifar-10-batches-py/data_batch_5'
test_file   = './cifar-10-batches-py/test_batch'

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

print "Loading database..."
train1 = unpickle(train_file1)
train2 = unpickle(train_file2)
train3 = unpickle(train_file3)
train4 = unpickle(train_file4)
train5 = unpickle(train_file5)
test   = unpickle(test_file)

x_train = np.concatenate((train1['data'],train2['data'],train3['data'],train4['data'],train5['data']))
y_train = np.concatenate((train1['labels'],train2['labels'],train3['labels'],train4['labels'],train5['labels'])).reshape(-1,1)
x_test = np.array(test['data'])
y_test = np.array(test['labels']).reshape(-1,1)

x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_test  = x_test.reshape(x_test.shape[0],32,32,3)

#normalize images
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255

print 'x_train shape:', x_train.shape

print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'


#training parameters
batch_size = 128
epochs     = 10

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

# create cnn model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# print cnn layers
print 'Network structure ----------------------------------'
for i, layer in enumerate(model.layers):
	print(i,layer.name)
	if hasattr(layer, 'output_shape'):
		print(layer.output_shape)
print '----------------------------------------------------'

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]

#print model.predict_classes(x_test) #classes predicted
#print model.predict_proba(x_test) #classes probability

pred = []
y_pred = model.predict_classes(x_test)
for i in range(len(x_test)):
	pred.append(y_pred[i])
    

### save for the confusion matrix
label = []
for i in range(len(x_test)):
	label.append(y_test[i][0])
print(confusion_matrix(label, pred))
