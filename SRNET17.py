# Larger CNN for the MNIST Dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from collections import Counter
from tensorflow.keras.layers import Input
import re, os, csv, math, operator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

#Contains 86 elements (Without Noble elements as it does not forms compounds in normal condition)
elements = ['H','Li','Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
            'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

# import training data 
def load_data(csvname):
    # load in data
    data = np.asarray(pd.read_csv(csvname))

    # import data and reshape appropriately
    X = data[:,0:-1]
    y = data[:,-1]
    y.shape = (len(y),1)
    
    return X,y

def convert(lst): 
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)} 
    return res_dct 

separate = re.compile('[A-Z][a-z]?|\d+\.\d')

def correction(x_train):
    new_x = []
    for i in range (0,x_train.shape[0]):
        new_x.append(separate.findall(x_train[i][0])) 
    new_x = np.asarray(new_x)
    new_x.shape = (len(new_x),1)    
    dict_x = convert(new_x[0][0])
    input_x = []
    for i in range (0,new_x.shape[0]):
        input_x.append(convert(new_x[i][0]))
        
    in_elements = np.zeros(shape=(len(input_x), len(elements)))
    comp_no = 0

    for compound in input_x:
        keys = compound.keys()
        for key in keys:
            in_elements[comp_no][elements.index(key)] = compound[key]
        comp_no+=1  

    data = in_elements    
    
    return data

# load data
x_train, y_train = load_data('dataset/train_set.csv')
x_test, y_test = load_data('dataset/test_set.csv')

new_x_train = correction(x_train)
new_x_test = correction(x_test)

new_y_train = y_train
new_y_test = y_test

new_y_train.shape = (len(new_y_train),)
new_y_test.shape = (len(new_y_test),)

batch_size1 = new_x_train.shape[0]
num_input1 = new_x_train.shape[1]

#in_chem = Input(shape=(num_input,))

# create model

in_layer = Input(shape=(86,))

layer_1 = Dense(1024)(visible)
layer_1 = BatchNormalization()(layer_1)
layer_1 = Activation('relu')(layer_1)

layer_2 = Dense(1024)(layer_1)
layer_2 = BatchNormalization()(layer_2)
layer_2 = Activation('relu')(layer_2)

layer_3 = Dense(1024)(layer_2)
layer_3 = BatchNormalization()(layer_3)
layer_3 = Activation('relu')(layer_3)

layer_4 = Dense(1024)(layer_3)
layer_4 = BatchNormalization()(layer_4)
layer_4 = Activation('relu')(layer_4)

gsk_1 = concatenate([in_layer, layer_4])

layer_5 = Dense(512)(gsk_1)
layer_5 = BatchNormalization()(layer_5)
layer_5 = Activation('relu')(layer_5)

layer_6 = Dense(512)(layer_5)
layer_6 = BatchNormalization()(layer_6)
layer_6 = Activation('relu')(layer_6)

layer_7 = Dense(512)(layer_6)
layer_7 = BatchNormalization()(layer_7)
layer_7 = Activation('relu')(layer_7)

gsk_2 = concatenate([gsk_1, layer_7])

layer_8 = Dense(256)(gsk_2)
layer_8 = BatchNormalization()(layer_8)
layer_8 = Activation('relu')(layer_8)

layer_9 = Dense(256)(layer_8)
layer_9 = BatchNormalization()(layer_9)
layer_9 = Activation('relu')(layer_9)

layer_10 = Dense(256)(layer_9)
layer_10 = BatchNormalization()(layer_10)
layer_10 = Activation('relu')(layer_10)

gsk_3 = concatenate([gsk_2, layer_10])

layer_11 = Dense(128)(gsk_3)
layer_11 = BatchNormalization()(layer_11)
layer_11 = Activation('relu')(layer_11)

layer_12 = Dense(128)(layer_11)
layer_12 = BatchNormalization()(layer_12)
layer_12 = Activation('relu')(layer_12)

layer_13 = Dense(128)(layer_12)
layer_13 = BatchNormalization()(layer_13)
layer_13 = Activation('relu')(layer_13)

gsk_4 = concatenate([gsk_3, layer_13])

layer_14 = Dense(64)(gsk_4)
layer_14 = BatchNormalization()(layer_14)
layer_14 = Activation('relu')(layer_14)

layer_15 = Dense(64)(layer_14)
layer_15 = BatchNormalization()(layer_15)
layer_15 = Activation('relu')(layer_15)

gsk_5 = concatenate([gsk_4, layer_15])

layer_16 = Dense(32)(gsk_5)
layer_16 = BatchNormalization()(layer_16)
layer_16 = Activation('relu')(layer_16)

gsk_6 = concatenate([gsk_5, layer_16])

out_layer = Dense(1)(gsk_6)

model = Model(inputs=in_layer, outputs=out_layer)

# Compile model
adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# Fit the model
model.fit(new_x_train, new_y_train,verbose=2, validation_data=(new_x_test, new_y_test), epochs=1000, batch_size=32, callbacks=[es])
y_predict = model.predict(new_x_test)
print(y_predict)
model.save_weights("model2.h5")