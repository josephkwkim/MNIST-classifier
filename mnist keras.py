###Load Data and Create Model###
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical

###Plot Results###
import matplotlib.pyplot as plt

####Bypass Warnings###
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_flat_data():
    # load data
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape(60000,784)
    train_x = train_x.astype('float32')
    train_x /= 255
    test_x = test_x.reshape(10000,784)
    test_x = test_x.astype('float32')
    test_x /= 255
    train_y = to_categorical(train_y,num_classes=10)
    test_y = to_categorical(test_y,num_classes=10)

    # check dimensions
    assert (np.shape(train_x) == (60000, 784))
    assert (np.shape(train_y) == (60000, 10))
    assert (np.shape(test_x) == (10000, 784))
    assert (np.shape(test_y) == (10000, 10))

    return train_x,train_y,test_x,test_y

def one_hidden_layer(epochs):
    #load data
    train_x,train_y,test_x,test_y = load_flat_data()

    #create model
    model = Sequential()
    model.add(Dense(100,input_dim=784,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation="softmax"))
    sgd = SGD(momentum=0.9,decay=0.01)
    model.compile(optimizer="sgd",loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    history = model.fit(train_x,train_y,verbose=2,batch_size=10,
                        validation_data=(test_x,test_y),epochs=epochs)

    #plot accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

def load_square_data():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = np.reshape(train_x.astype('float32')/255,(60000,28,28,1))
    test_x = np.reshape(test_x.astype('float32')/255,(10000,28,28,1))
    train_y = to_categorical(train_y,10)
    test_y = to_categorical(test_y,10)

    #check shape
    assert (np.shape(train_x)==(60000,28,28,1))
    assert (np.shape(test_x)==(10000,28,28,1))
    assert (np.shape(train_y)==(60000,10))
    assert (np.shape(test_y)==(10000,10))

    return train_x,train_y,test_x,test_y

def load_square_kaggle_data():
    flattened = pd.read_csv('mnist_test_data.csv')
    test_x,test_y = [],[]
    for (index,row) in flattened.iterrows():
        test_x.append(np.reshape(row.values,(28,28)))
        test_y.append(index)

    return np.reshape(np.array(test_x),(28000,28,28,1)),np.array(test_y)

def convolutional(epochs,kaggle=False):
    train_x,train_y,test_x,test_y = load_square_data()
    if kaggle:
        val_x,val_y = load_square_kaggle_data()

    model = Sequential()
    model.add(Conv2D(filters=20,kernel_size=(5,5),input_shape=(28,28,1),
                     activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=15,kernel_size=(4,4),activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    sgd = SGD(momentum=0.9,nesterov=True)
    model.compile(optimizer='sgd',loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    print (model.summary())

    history = model.fit(train_x, train_y, verbose=2,
                        validation_data=(test_x, test_y), epochs=epochs)

    # plot accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    if kaggle:
        vectorized = model.predict(val_x/255)

        answers = []
        for vector in vectorized:
            answers.append(np.argmax(vector))

        df = pd.DataFrame()

        df['ImageId'] = val_y
        df['Label'] = np.array(answers)

        df.to_csv('mnist.csv',index=False,columns=['ImageId','Label'])

        return model



