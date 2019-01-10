#!/usr/bin/env python

__author__ = "Chaitanya"

#import all the dependencies

import tensorflow as tf
with tf.device('/device:GPU:2'):
    import keras
    from keras.models import Sequential
    #from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
    from keras import backend as K
    import pickle as pk
    batch_size = 1024
    epochs = 50
    import numpy as  np
    import os
    cwd = os.getcwd()
    #x_train = os.path.join(cwd,'Noisy_TCDTIMIT/Babble/20/volunteers/01M/straightcam')
    #y_train = os.path.join(cwd,'Clean/volunteers/01M/straightcam')

    pickle_train = open("noiseBabble_xtrain.pickle","rb")
    x_train = pk.load(pickle_train)
    pickle_trainlabel = open("cleanBabble.pickle","rb")
    y_train = pk.load(pickle_trainlabel)

    print("Got the input data")
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_train = x_train.reshape(x_train.shape[0], 129, 16, 1)
    y_train = y_train.reshape(y_train.shape[0], 129, 16, 1)
    print(np.asarray(x_train).shape)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7),activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',input_shape=(129,16,1)))
    model.add(Conv2D(128, (5, 5), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(Conv2D(256, (1, 1), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    #model.add(Conv2DTranspose(128, (3, 3), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(Conv2DTranspose(128, (3, 3), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(Conv2DTranspose(64, (5, 5), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(Conv2DTranspose(1, (7, 7), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    print(model.summary())

    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(np.asarray(x_train), np.asarray(y_train),
              batch_size=batch_size,
              epochs=epochs,
              validation_split = 0.01
             )

    model.save_weights('weights_1.pkl')

    #score = model.evaluate(x_train, y_train, verbose=0)

    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
