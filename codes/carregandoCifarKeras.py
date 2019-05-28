from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers import  MaxPooling2D
import keras.utils


input_shape = (32,32,3)

model = Sequential()

#Camada1
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
#Camada2
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
#Camada3
model.add(Conv2D(32, (3, 3), activation='relu'))
#Camada4
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))

#camadas totalmente conectadas
model.add(Flatten()) #alinhar os dados em um vetor único
model.add(Dense(units=256))
model.add(Dropout(0.5))
model.add(Dense(units=256))

#Classificador final
model.add(Dense(10,activation="softmax"))

model.summary()



adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.99)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc', 'mae'])



from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(y_train)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.fit(x=x_train,
          y=y_train,
          batch_size=64,
          epochs=5,
          validation_data=(x_test, y_test))
