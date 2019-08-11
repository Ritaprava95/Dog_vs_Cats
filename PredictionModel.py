# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:29:26 2019

@author: ritap
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.optimizers import SGD


def use_gpu():
    # Creates a graph.
    with tf.device('/gpu:0'):
      a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
      b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))
    print("Using The GPU")

use_gpu()
   
X = np.load('train_X.npy')
y = np.load('train_y.npy')
X = X/255
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=21)

#first model is a sequential one
def model():   
    images = Input((100, 100, 3))
    X = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(images)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.2)(X)
    X = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.2)(X)
    X = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.2)(X)
    X = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.2)(X)
    X = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu', kernel_constraint=maxnorm(3))(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation='sigmoid')(X)
    # Compile model
    lrate = 0.001
    #decay = lrate/epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model

model = model()

model.fit(train_X, train_y, epochs=50, batch_size=32)

#let's try resnet
def identity_block(X, f, filters, stage, block):
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer = 'he_uniform', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', kernel_initializer = 'he_uniform', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer = 'he_uniform', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
    


def convolutional_block(X, f, s, filters, stage, block):
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), padding = 'valid', kernel_initializer = 'he_uniform', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', kernel_initializer = 'he_uniform', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer = 'he_uniform', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    
    X_shortcut = Conv2D(filters = F3, kernel_size = (f,f), strides = (s,s), padding = 'same', kernel_initializer = 'he_uniform', name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    X_shortcut = Activation('relu')(X_shortcut)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = 'he_uniform')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


model = ResNet50(input_shape = (100, 100, 3), classes = 1 )
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_y, validation_split=0.2, epochs=50, batch_size=32)
