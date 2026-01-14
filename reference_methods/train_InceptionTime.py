#!/usr/bin/env python
# coding: utf-8

# In[2]:


## load datasets
import numpy as np
import pandas as pd

CLASSES = ["Bull", "Bear"]
LABEL_BULL = CLASSES.index("Bull")
LABEL_BEAR = CLASSES.index("Bear")

## load datasets from disk
datasets = np.load("datasets.npz")
x_train, y_train = datasets["x_train"], datasets["y_train"]
x_valid, y_valid = datasets["x_valid"], datasets["y_valid"]
x_test, y_test = datasets["x_test"], datasets["y_test"]

label_distribution = pd.DataFrame([{"DataSet" : "train",
                                    "Bull" : np.count_nonzero(y_train == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_train == LABEL_BEAR)},
                                   {"DataSet" : "valid",
                                    "Bull" : np.count_nonzero(y_valid == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_valid == LABEL_BEAR)},
                                   {"DataSet" : "test",
                                    "Bull" : np.count_nonzero(y_test == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_test == LABEL_BEAR)}])
label_distribution


# In[5]:


## construct model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, \
                                    BatchNormalization, Activation, \
                                    Add, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

def inception_model(input_tensor):
    bottleneck = Conv1D(filters=32, kernel_size=1, padding="same", activation=None, use_bias=False)(input_tensor)
    conv3 = Conv1D(filters=32, kernel_size=3, padding="same", activation=None, use_bias=False)(bottleneck)
    conv5 = Conv1D(filters=32, kernel_size=5, padding="same", activation=None, use_bias=False)(bottleneck)
    conv7 = Conv1D(filters=32, kernel_size=7, padding="same", activation=None, use_bias=False)(bottleneck)
    mp = MaxPooling1D(pool_size=3, strides=1, padding="same")(input_tensor)
    mpbottleneck = Conv1D(filters=32, kernel_size=1, padding="same", activation=None, use_bias=False)(mp)
    x = Concatenate(axis=-1)([conv3, conv5, conv7, mpbottleneck])
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def shortcut_layer(input_tensor1, input_tensor2):
    shortcut = Conv1D(filters=input_tensor2.shape[-1], kernel_size=1, padding="same", activation=None, use_bias=False)(input_tensor1)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, input_tensor2])
    x = Activation("relu")(x)
    return x


## build model
n_time_steps = x_train.shape[1] # 100 days in time-window
n_features = x_train.shape[2] # 5 features: open, close, max, min, volume
# create Input
input_layer = Input(shape=(n_time_steps, n_features))
x = input_layer
input_residual = input_layer

for i in range(6):
    x = inception_model(x)
    if i % 3 == 2:
        x = shortcut_layer(input_residual, x)
        input_residual = x

x = GlobalAveragePooling1D()(x)
output_layer = Dense(len(CLASSES), activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.summary()


# In[1]:


## train model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model_checkpoint = ModelCheckpoint(filepath="best_model_hdf5.keras", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)
callbacks = [model_checkpoint, early_stopping]

train_history = model.fit(x_train, to_categorical(y_train),
                          validation_data=(x_valid, to_categorical(y_valid)),
                          batch_size=2048, epochs=1000, callbacks=callbacks)


# In[4]:


## plot training
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
axes[0].set_title("Loss")
axes[0].set_yscale("log")
axes[0].plot(train_history.history["loss"], label="Training")
axes[0].plot(train_history.history["val_loss"], label="Validation")
axes[0].legend()

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].plot(train_history.history["accuracy"], label="Training")
axes[1].plot(train_history.history["val_accuracy"], label="Validation")
axes[1].legend()


# In[ ]:




