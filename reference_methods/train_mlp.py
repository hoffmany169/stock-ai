#!/usr/bin/env python
# coding: utf-8

# In[1]:


## train_mlp: training multilayer
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


# In[8]:


type(y_train)


# In[3]:


## construct multilayer perception model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

# build model
n_time_steps = x_train.shape[1] # 100 days in time-window
n_features = x_train.shape[2] # 5 features: open, close, max, min, volume
# create Input
input_layer = Input(shape=(n_time_steps, n_features))
x = Flatten()(input_layer)
x = Dense(256, activation="relu")(x)
x = Dense(256, activation="relu")(x)
output_layer = Dense(len(CLASSES), activation="softmax")(x)
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()


# In[2]:


x_train.shape


# In[4]:


1709*100*5


# In[9]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model_checkpoint = ModelCheckpoint(filepath="best_model_hdf5.keras", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)
callbacks = [model_checkpoint, early_stopping]

train_history = model.fit(x_train, to_categorical(y_train),
                          validation_data=(x_valid, to_categorical(y_valid)),
                          batch_size=2048, epochs=1000, callbacks=callbacks)


# In[11]:


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




