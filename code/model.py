#!/usr/bin/env python
# coding: utf-8

from keras.layers import Input, Embedding, LSTM, Dense, concatenate,Bidirectional, Dropout, TimeDistributed
from keras.models import Model
from keras import optimizers
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import preprocess as pre
import numpy as np



def create_keras_model_parallel(vocab_size_uni,vocab_size_bi, char_embedding_size,
                                bigram_embedding_size, hidden_size,input_dropout,lstm_dropout):
    print("Creating KERAS model")
    

    input_uni = Input(shape=(None,))
    input_bi = Input(shape=(None,))
    
    
    x1 = Embedding(vocab_size_uni,char_embedding_size,mask_zero = True)(input_uni)
    x2 = Embedding(vocab_size_bi,bigram_embedding_size,mask_zero = True)(input_bi)
    
    drp1 = Dropout(input_dropout)(x1)
    drp2 = Dropout(input_dropout)(x2)
    
    concat_input = concatenate([drp1,drp2],name="concat")
    

    bidi = Bidirectional(LSTM(hidden_size,name="Bidirectional_lstm",return_sequences=True,recurrent_dropout=lstm_dropout,dropout=lstm_dropout))(concat_input)

    
    output = TimeDistributed(Dense(4,activation='softmax'))(bidi)
    model = Model(inputs=[input_uni,input_bi],outputs=output)
    return model


def batch_generator(x_uni, x_bi, Y, batch_size, shuffle=False):
    while True:
      if not shuffle:
          for start in range(0, len(x_uni), batch_size):
              end = start + batch_size
              max_length = pre.find_max_length(x_uni[start:end])
              a = x_uni[start:end]
              b = x_bi[start:end]
              c = Y[start:end]
              a = pad_sequences(a,maxlen=max_length,padding='post')
              b = pad_sequences(b,maxlen=max_length,padding='post')
              c = pad_sequences(c,maxlen=max_length,padding='post')
              c = pre.convert_to_categorical(c)
              yield [a,b], c
      else:
          perm = np.random.permutation(len(x_uni))
          for start in range(0, len(x_uni), batch_size):
              end = start + batch_size
              max_length = pre.find_max_length(x_uni[perm[start:end]])
              a = x_uni[perm[start:end]]
              b = x_bi[perm[start:end]]
              c = Y[perm[start:end]]
              a = pad_sequences(a,maxlen=max_length,padding='post')
              b = pad_sequences(b,maxlen=max_length,padding='post')
              c = pad_sequences(c,maxlen=max_length,padding='post')
              c = pre.convert_to_categorical(c)
              yield [a,b], c


def compile_keras_model(model,optimizer,learning_rate):
    if(str.lower(optimizer) == "sgd"):
      opt = optimizers.SGD(lr=learning_rate, clipnorm=0.1 , momentum=0.95, nesterov=True)
    elif(str.lower(optimizer) == "adam"):
      opt = optimizers.Adam(lr=learning_rate,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    return model

def train_keras_model(model,train_x_uni,train_x_bi,train_y_uni,
                      dev_x_uni,dev_x_bi,dev_y_uni,
                      batch_size,epochs,steps_per_epochs,validation_steps):
    '''
    Method used to train the model
    :model the model to train
    :train_x_uni the unigrams input to train
    :train_x_bi the bigrams input to train
    :train_y_uni the label of the training
    :dev_x_uni the unigrams input to test
    :dev_x_bi the bigrams input to test
    :dev_y_uni the label of the test
    :batch_size 
    :epochs
    :steps_per_epochs
    :validation_steps
    :return the statistics about the model performance
    '''
    early_stopping = EarlyStopping(monitor="val_loss",patience=2)
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    cbk = [early_stopping,checkpointer]

    print("\nStarting training...")
    stats = model.fit_generator(batch_generator(train_x_uni,train_x_bi,train_y_uni,batch_size,shuffle=False),
                        steps_per_epoch=steps_per_epochs,
                        epochs=epochs,
                        callbacks=cbk,
                        verbose = 1,
                        validation_data=batch_generator(dev_x_uni,dev_x_bi,dev_y_uni,batch_size,shuffle=False),
                        validation_steps=validation_steps)
    print("Training complete.\n")
    return stats

