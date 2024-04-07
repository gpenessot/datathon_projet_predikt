# %%
#!pip install vitaldb
#!pip install imbalanced-learn


# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input, BatchNormalization, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from keras.optimizers import Adam 

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# 

# %%
# Load the numpy file
datax = np.load('x_70.npy',allow_pickle=True)
datay = np.load('y_70.npy',allow_pickle=True)
# Access the data
print(datax.shape)
print(datay.shape)

datax=datax[:,:,0]
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, Y_resampled = smote.fit_resample(datax[:250000], datay[:250000])

#X_resampled, Y_resampled = datax[:250000], datay[:250000]

train_x_valid, test_x_valid, train_y_valid, test_y_valid = train_test_split(X_resampled, 
                                                                            Y_resampled, 
                                                                            test_size=0.2, 
                                                                            random_state=42,
                                                                            stratify=Y_resampled)



# %%
# add axis for CNN
train_x_valid = train_x_valid[...,None]  
# add axis for CNN
test_x_valid = test_x_valid[...,None]  

train_y_valid=train_y_valid.astype(int)

# %%
LSTM_NODES = 128
BATCH_SIZE = 256

# making output folder
tempdir = 'output'
if not os.path.exists(tempdir):
    os.mkdir(tempdir)
weight_path = tempdir + "/weights_70.h5"

# build a model
model = Sequential()
model.add(LSTM(LSTM_NODES,
               return_sequences=True, 
               input_shape=train_x_valid.shape[1:]))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Initialize the model
# model = Sequential()
# # Add an LSTM layer with more units and return_sequences=True for sequence output
# model.add(LSTM(units=128, 
#                return_sequences=True, 
#                #input_shape=train_x_valid.shape[1:]
#                ))
# # Add another LSTM layer with more units and return_sequences=True
# model.add(LSTM(units=64, return_sequences=True))
# # Add a Bidirectional LSTM layer for more complex learning
# model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
# # Add a dropout layer for regularization
# model.add(Dropout(0.5))
# # Add a Dense layer for final output
# model.add(Dense(units=1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=1e-3), 
              metrics=['accuracy', 
                       tf.keras.metrics.AUC()])

hist = model.fit(train_x_valid, 
                 train_y_valid, 
                 validation_split=0.3, 
                 epochs=50, 
                 batch_size=BATCH_SIZE,
                 callbacks=[ModelCheckpoint(monitor='val_loss', filepath=weight_path, verbose=1, save_best_only=True),
                            EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
                            ])

# reload the best model
model.load_weights(weight_path)
open(tempdir + "/model_70.json", "wt").write(model.to_json())

# %%
import matplotlib.pyplot as plt

# Plot training & validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('{}/loss_plot_70.png'.format(tempdir))
plt.show()


# %%
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve

# make prediction on the test dataset
test_y_pred = model.predict(test_x_valid).flatten()

precision, recall, thmbps = precision_recall_curve(test_y_valid, test_y_pred)
auprc = auc(recall, precision)

fpr, tpr, thmbps = roc_curve(test_y_valid, test_y_pred)
auroc = auc(fpr, tpr)

thval = 0.5
f1 = f1_score(test_y_valid, test_y_pred > thval)
acc = accuracy_score(test_y_valid, test_y_pred > thval)
tn, fp, fn, tp = confusion_matrix(test_y_valid, test_y_pred > thval).ravel()

testres = 'auroc={:.3f}, auprc={:.3f} acc={:.3f}, F1={:.3f}, PPV={:.1f}, NPV={:.1f}, TN={}, fp={}, fn={}, TP={}'.format(auroc, auprc, acc, f1, tp/(tp+fp)*100, tn/(tn+fn)*100, tn, fp, fn, tp)
print(testres)

# auroc curve
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('{}/auroc_70.png'.format(tempdir))
plt.close()

# auprc curve
plt.figure(figsize=(10, 10))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('{}/auprc_70.png'.format(tempdir))
plt.close()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming you have calculated the confusion matrix
conf_matrix = confusion_matrix(test_y_valid, test_y_pred > thval)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('{}/confusion_matrix_70.png'.format(tempdir))
plt.show()


# %%



