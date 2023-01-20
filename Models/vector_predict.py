# Tensorflow/ Keras code for a vector-prediction model
# Objective was to predict the order of arrival of competitors in a race.
# Since only the order was important, not the magnitude, using 
# CosineSimilarity as a loss fn provided better results.

# Code contains a custom DataGenerator that organizes the 
# races into batches with padding + masking.



# TO DO: 
# add simulated data


#import mysql.connector as connector
from datetime import datetime
import pandas as pd
import numpy as np

import os
import shutil
import pickle
import copy

# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import make_pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# from sklearn.compose import make_column_transformer
# from sklearn.pipeline import make_union
# from sklearn.compose import make_column_selector
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.metrics import precision_score
from tensorflow.keras import metrics


import pandas; print(pandas.__version__)
import numpy; print(numpy.__version__)
import sklearn; print(sklearn.__version__)
import tensorflow; print(tensorflow.__version__)
import keras; print(keras.__version__)





# %%

# Load data (cannot be shared publicly)
#df_for_grouping = pd.read_csv()
#df_for_grouping_test = pd.read_csv()

# %%
n_competitors = 14 # output vector size (the number of competitors in each race)
# NB: Use padding, masking if too few competitors, use the first n_competitors if there are too many
# (Data had the competitors already sorted by order within each group, so dropped the losers, if any)

# %%

# What we're trying to predict: a column labelled "y_col_to_predict"
# df_for_grouping_test.y_col_to_predict.value_counts() 
# order that the competitors arrived in, a permutation of (1, ..., n_competitors)




# %%
def val_top_K_precision(prediction, y_true, threshold=3):
    # The client wanted this custom metric to see if we correctly 
    # predicted the top e.g. 3 horses
    pred = prediction.numpy().argsort() # converts a float prediction into the ranking like y_col_to_predict (0-indexed)
    true = y_true.numpy().argsort()
    pred =  (pred < threshold).astype(int) 
    true = (true < threshold).astype(int)
    m = tf.keras.metrics.Precision()
    m.update_state(true, pred)
    return(m.result().numpy())

# %%

# %%
dropout = 0.3
PATIENCE = 10
nb_epoch = 1 # 100
batch_size = 16
ncolumns_to_use = 98 # only use the first e.g. 98 columns of the X data

from tensorflow.keras.layers import Dense,Masking,Flatten
input_shape = (n_competitors, ncolumns_to_use)
model = Sequential()
model.add(Masking(mask_value=-1,input_shape=input_shape ))
model.add(layers.Dense(1024, activation='relu')) 
model.add(Dropout(dropout))
model.add(layers.Dense(512, activation='relu')) 
model.add(Dropout(dropout))
model.add(layers.Dense(256, activation='relu')) 
model.add(Dropout(dropout))
model.add(layers.Dense(128, activation='relu')) 
model.add(Dropout(dropout))
model.add(layers.Dense(64, activation='relu')) 
model.add(Dropout(dropout))
model.add(layers.Dense(32, activation='relu')) 
model.add(Dropout(dropout))
model.add(Flatten())
model.add(layers.Dense(n_competitors, activation = 'sigmoid')) # output vector: dim is n_competitors


es = EarlyStopping(monitor="val_top_K_precision", 
                   mode='max',
                   patience=PATIENCE, 
                   restore_best_weights=True)


model.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity(), 
   metrics=[meilleur_K_precision])
# model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())#, 


# %%
print(model.summary())

# %%
def pad_y(y, n_competitors=14):
    new_y = copy.deepcopy(y)
    l = len(y)
    if l < n_competitors:
        to_append = pd.Series([n_competitors for x in range(n_competitors-l)] )
        return(pd.concat([new_y, to_append]))
    else:
        return(y[:n_competitors])

# %%
def pad_tensor_to_dim(x, n_competitors=14):
    curr_n_compet = x.shape[0]
    if curr_n_compet == n_competitors: # already the right size
        return(x)
    elif curr_n_compet > n_competitors: # if too many competitors, only use the first e.g. 14 (in data, losers were last)
        return(x[:n_competitors,:])
    else:
        paddings = tf.constant([[0, n_competitors-curr_n_compet,], [0, 0]]) # pad the difference e.g. 14-9= 5
        return(tf.pad(x, paddings, "CONSTANT"))


# %%
class DataGenerator(keras.utils.Sequence):
    'Create data for vector prediction. Batch is selected as a subset based on race_id'
    def __init__(self, X, batch_size=16, shuffle=True, X_only=False, ncolumns_to_use=98):
        'Initialization'
        self.X = X
        self.n_competitors = n_competitors
        self.dim = (n_competitors, ncolumns_to_use) # only use the first 98 columns
        self.races = list(set(X.race_id))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.X_only = X_only
        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.races) / self.batch_size))

    def __getitem__(self, index):
        'Create one batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_races_temp = [self.races[k] for k in indexes]
        X, y = self.__data_generation(list_races_temp)
        return(X, y)


    def on_epoch_end(self):
        'Update index after each epoch'
        self.indexes = np.arange(len(self.races))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __get_one_race(self, race):
        'subset dataframe by race'
        Xrel = self.X[self.X.race_id==race ]
        if self.shuffle == True:
            Xrel = Xrel.sample(frac=1) # randomly sort...
        #Xrel = Xrel.replace(np.NaN, 0) # replace Nan with 0 -> masking
        curr_x = Xrel.iloc[:,:ncolumns_to_use]
        if self.X_only:
            curr_y = np.zeros(self.n_competitors)
        else:
            curr_y = Xrel['y_col_to_predict']
        return(curr_x, curr_y)
        
    def __data_generation(self, list_races_temp):
        'Generate batchs and pad to batch_size'
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_competitors ))

        for i, race in enumerate(list_races_temp):
            currx, curry = self.__get_one_race(race)
            currx1 = tf.convert_to_tensor(currx)
            if currx1.shape[0]<n_competitors:
                currx1 = pad_tensor_to_dim(currx1, n_competitors)
            elif currx1.shape[0] > n_competitors:
                currx1 = currx1[:n_competitors,:]     
            curry1 = pad_y(curry, n_competitors)

            X[i,] = currx1
            y[i,] = curry1

        return X, y


# %%

# Use only first 98 columns - ncolumns_to_use
training_generator = DataGenerator(df_for_grouping, batch_size=16, shuffle=True, ncolumns_to_use=98)
validation_generator = DataGenerator(df_for_grouping_test, batch_size=16, shuffle=False)
prediction_generator = DataGenerator(df_for_grouping_test, batch_size=16, shuffle=False, X_only=True)


# %%
%load_ext tensorboard

# %%
tensorboard --logdir=logs/fit


# %%
tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1, )

# %%
#from keras.callbacks import ModelCheckpoint

# checkpoint_filepath = "tmp_checkpoint.hdf5"
# checkpoint = ModelCheckpoint(filepath= checkpoint_filepath,
#                              monitor='loss',
#                              save_weights_only=True, 
#                              save_freq=100,
#                              save_best_only=True)

# %%

tf.config.run_functions_eagerly(True)


# %%
history = model.fit(training_generator,
                    validation_data=validation_generator,
                    batch_size=batch_size,
                    epochs=10,
                    callbacks=[tensorboard_callback, es], #,checkpoint
                    steps_per_epoch=100)



# %%
#model.load_weights(checkpoint_filepath)
