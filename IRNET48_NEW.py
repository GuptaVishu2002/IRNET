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
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
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

layer_1 = Dense(1024)(in_layer)
layer_1 = BatchNormalization()(layer_1)
layer_1 = Activation('relu')(layer_1)

#gsk_1 = concatenate([in_layer, layer_1])

layer_2 = Dense(1024)(layer_1)
layer_2 = BatchNormalization()(layer_2)
layer_2 = Activation('relu')(layer_2)

gsk_2 = add([layer_1, layer_2])

layer_3 = Dense(1024)(gsk_2)
layer_3 = BatchNormalization()(layer_3)
layer_3 = Activation('relu')(layer_3)

gsk_3 = add([gsk_2, layer_3])

layer_4 = Dense(1024)(gsk_3)
layer_4 = BatchNormalization()(layer_4)
layer_4 = Activation('relu')(layer_4)

gsk_4 = add([gsk_3, layer_4])

layer_5 = Dense(1024)(gsk_4)
layer_5 = BatchNormalization()(layer_5)
layer_5 = Activation('relu')(layer_5)

gsk_5 = add([gsk_4, layer_5])

layer_6 = Dense(1024)(gsk_5)
layer_6 = BatchNormalization()(layer_6)
layer_6 = Activation('relu')(layer_6)

gsk_6 = add([gsk_5, layer_6])

layer_7 = Dense(1024)(gsk_6)
layer_7 = BatchNormalization()(layer_7)
layer_7 = Activation('relu')(layer_7)

gsk_7 = add([gsk_6, layer_7])

layer_8 = Dense(1024)(gsk_7)
layer_8 = BatchNormalization()(layer_8)
layer_8 = Activation('relu')(layer_8)

gsk_8 = add([gsk_7, layer_8])

layer_9 = Dense(512)(gsk_8)
layer_9 = BatchNormalization()(layer_9)
layer_9 = Activation('relu')(layer_9)

#gsk_9 = concatenate([gsk_8, layer_9])

layer_10 = Dense(512)(layer_9)
layer_10 = BatchNormalization()(layer_10)
layer_10 = Activation('relu')(layer_10)

gsk_10 = add([layer_9, layer_10])

layer_11 = Dense(512)(gsk_10)
layer_11 = BatchNormalization()(layer_11)
layer_11 = Activation('relu')(layer_11)

gsk_11 = add([gsk_10, layer_11])

layer_12 = Dense(512)(gsk_11)
layer_12 = BatchNormalization()(layer_12)
layer_12 = Activation('relu')(layer_12)

gsk_12 = add([gsk_11, layer_12])

layer_13 = Dense(512)(gsk_12)
layer_13 = BatchNormalization()(layer_13)
layer_13 = Activation('relu')(layer_13)

gsk_13 = add([gsk_12, layer_13])

layer_14 = Dense(512)(gsk_13)
layer_14 = BatchNormalization()(layer_14)
layer_14 = Activation('relu')(layer_14)

gsk_14 = add([gsk_13, layer_14])

layer_15 = Dense(512)(gsk_14)
layer_15 = BatchNormalization()(layer_15)
layer_15 = Activation('relu')(layer_15)

gsk_15 = add([gsk_14, layer_15])

layer_16 = Dense(512)(gsk_15)
layer_16 = BatchNormalization()(layer_16)
layer_16 = Activation('relu')(layer_16)

gsk_16 = add([gsk_15, layer_16])

layer_17 = Dense(256)(gsk_16)
layer_17 = BatchNormalization()(layer_17)
layer_17 = Activation('relu')(layer_17)

#gsk_17 = concatenate([gsk_16, layer_17])

layer_18 = Dense(256)(layer_17)
layer_18 = BatchNormalization()(layer_18)
layer_18 = Activation('relu')(layer_18)

gsk_18 = add([layer_17, layer_18])

layer_19 = Dense(256)(gsk_18)
layer_19 = BatchNormalization()(layer_19)
layer_19 = Activation('relu')(layer_19)

gsk_19 = add([gsk_18, layer_19])

layer_20 = Dense(256)(gsk_19)
layer_20 = BatchNormalization()(layer_20)
layer_20 = Activation('relu')(layer_20)

gsk_20 = add([gsk_19, layer_20])

layer_21 = Dense(256)(gsk_20)
layer_21 = BatchNormalization()(layer_21)
layer_21 = Activation('relu')(layer_21)

gsk_21 = add([gsk_20, layer_21])

layer_22 = Dense(256)(gsk_21)
layer_22 = BatchNormalization()(layer_22)
layer_22 = Activation('relu')(layer_22)

gsk_22 = add([gsk_21, layer_22])

layer_23 = Dense(256)(gsk_22)
layer_23 = BatchNormalization()(layer_23)
layer_23 = Activation('relu')(layer_23)

gsk_23 = add([gsk_22, layer_23])

layer_24 = Dense(256)(gsk_23)
layer_24 = BatchNormalization()(layer_24)
layer_24 = Activation('relu')(layer_24)

gsk_24 = add([gsk_23, layer_24])

layer_25 = Dense(128)(gsk_24)
layer_25 = BatchNormalization()(layer_25)
layer_25 = Activation('relu')(layer_25)

#gsk_25 = concatenate([gsk_24, layer_25])

layer_26 = Dense(128)(layer_25)
layer_26 = BatchNormalization()(layer_26)
layer_26 = Activation('relu')(layer_26)

gsk_26 = add([layer_25, layer_26])

layer_27 = Dense(128)(gsk_26)
layer_27 = BatchNormalization()(layer_27)
layer_27 = Activation('relu')(layer_27)

gsk_27 = add([gsk_26, layer_27])

layer_28 = Dense(128)(gsk_27)
layer_28 = BatchNormalization()(layer_28)
layer_28 = Activation('relu')(layer_28)

gsk_28 = add([gsk_27, layer_28])

layer_29 = Dense(128)(gsk_28)
layer_29 = BatchNormalization()(layer_29)
layer_29 = Activation('relu')(layer_29)

gsk_29 = add([gsk_28, layer_29])

layer_30 = Dense(128)(gsk_29)
layer_30 = BatchNormalization()(layer_30)
layer_30 = Activation('relu')(layer_30)

gsk_30 = add([gsk_29, layer_30])

layer_31 = Dense(128)(gsk_30)
layer_31 = BatchNormalization()(layer_31)
layer_31 = Activation('relu')(layer_31)

gsk_31 = add([gsk_30, layer_31])

layer_32 = Dense(128)(gsk_31)
layer_32 = BatchNormalization()(layer_32)
layer_32 = Activation('relu')(layer_32)

gsk_32 = add([gsk_31, layer_32])

layer_33 = Dense(64)(gsk_32)
layer_33 = BatchNormalization()(layer_33)
layer_33 = Activation('relu')(layer_33)

#gsk_33 = concatenate([gsk_32, layer_33])

layer_34 = Dense(64)(layer_33)
layer_34 = BatchNormalization()(layer_34)
layer_34 = Activation('relu')(layer_34)

gsk_34 = add([layer_33, layer_34])

layer_35 = Dense(64)(gsk_34)
layer_35 = BatchNormalization()(layer_35)
layer_35 = Activation('relu')(layer_35)

gsk_35 = add([gsk_34, layer_35])

layer_36 = Dense(64)(gsk_35)
layer_36 = BatchNormalization()(layer_36)
layer_36 = Activation('relu')(layer_36)

gsk_36 = add([gsk_35, layer_36])

layer_37 = Dense(64)(gsk_36)
layer_37 = BatchNormalization()(layer_37)
layer_37 = Activation('relu')(layer_37)

gsk_37 = add([gsk_36, layer_37])

layer_38 = Dense(64)(gsk_37)
layer_38 = BatchNormalization()(layer_38)
layer_38 = Activation('relu')(layer_38)

gsk_38 = add([gsk_37, layer_38])

layer_39 = Dense(64)(gsk_38)
layer_39 = BatchNormalization()(layer_39)
layer_39 = Activation('relu')(layer_39)

gsk_39 = add([gsk_38, layer_39])

layer_40 = Dense(64)(gsk_39)
layer_40 = BatchNormalization()(layer_40)
layer_40 = Activation('relu')(layer_40)

gsk_40 = add([gsk_39, layer_40])

layer_41 = Dense(64)(gsk_40)
layer_41 = BatchNormalization()(layer_41)
layer_41 = Activation('relu')(layer_41)

gsk_41 = add([gsk_40, layer_41])

layer_42 = Dense(32)(gsk_41)
layer_42 = BatchNormalization()(layer_42)
layer_42 = Activation('relu')(layer_42)

#gsk_42 = concatenate([gsk_41, layer_42])

layer_43 = Dense(32)(layer_42)
layer_43 = BatchNormalization()(layer_43)
layer_43 = Activation('relu')(layer_43)

gsk_43 = add([layer_42, layer_43])

layer_44 = Dense(32)(gsk_43)
layer_44 = BatchNormalization()(layer_44)
layer_44 = Activation('relu')(layer_44)

gsk_44 = add([gsk_43, layer_44])

layer_44 = Dense(32)(gsk_43)
layer_44 = BatchNormalization()(layer_44)
layer_44 = Activation('relu')(layer_44)

gsk_44 = add([gsk_43, layer_44])

layer_45 = Dense(16)(gsk_44)
layer_45 = BatchNormalization()(layer_45)
layer_45 = Activation('relu')(layer_45)

#gsk_45 = concatenate([gsk_44, layer_45])

layer_46 = Dense(16)(layer_45)
layer_46 = BatchNormalization()(layer_46)
layer_46 = Activation('relu')(layer_46)

gsk_46 = add([layer_45, layer_46])

layer_47 = Dense(16)(gsk_46)
layer_47 = BatchNormalization()(layer_47)
layer_47 = Activation('relu')(layer_47)

gsk_47 = add([gsk_46, layer_47])

out_layer = Dense(1)(gsk_47)

model = Model(inputs=in_layer, outputs=out_layer)

# Compile model
adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
# Fit the model
model.fit(new_x_train, new_y_train,verbose=2, validation_data=(new_x_test, new_y_test), epochs=1000, batch_size=32, callbacks=[es])
y_predict = model.predict(new_x_test)
f = open( 'resultIR48.txt', 'w' )
f.write(y_predict)
f.close()
model.save_weights("modelIR48.h5")