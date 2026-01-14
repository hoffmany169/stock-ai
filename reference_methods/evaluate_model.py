#!/usr/bin/env python
# coding: utf-8

# In[1]:


## install libraries
get_ipython().system('pip install seaborn')


# In[2]:


import numpy as np
import pandas as pd

CLASSES = ["Bull", "Bear"]
LABEL_BULL = CLASSES.index("Bull")
LABEL_BEAR = CLASSES.index("Bear")

## load datasets from disk
datasets = np.load("datasets.npz")
x_test, y_test = datasets["x_test"], datasets["y_test"]
label_distribution = pd.DataFrame([{"DataSet" : "test",
                                    "Bull" : np.count_nonzero(y_test == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_test == LABEL_BEAR)}])
label_distribution


# In[3]:


## load best model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

model = keras.models.load_model("best_model_hdf5.keras")


# In[4]:


## evaluate model
model.evaluate(x_test, to_categorical(y_test))


# In[5]:


## draw confusion matrix
from tensorflow.math import argmax, confusion_matrix

y_pred_prob = model.predict(x_test)
y_pred = argmax(y_pred_prob, axis=-1)
cm = confusion_matrix(y_test, y_pred, num_classes=len(CLASSES)).numpy()


# In[6]:


cm


# In[7]:


## plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 4))
sns.heatmap(cm, xticklabels=CLASSES, yticklabels=CLASSES, annot=True, fmt='g')
plt.xlabel("Predication")
plt.ylabel("Label")


# In[8]:


## calculate the rate of prediction
n_bull_pred = cm[LABEL_BULL, LABEL_BULL] + cm[LABEL_BEAR, LABEL_BULL]
n_bull_true_pos = cm[LABEL_BULL, LABEL_BULL]
bull_accuracy = n_bull_true_pos / n_bull_pred if n_bull_pred > 0 else 0

n_bear_pred = cm[LABEL_BEAR, LABEL_BEAR] + cm[LABEL_BULL, LABEL_BEAR]
n_bear_true_pos = cm[LABEL_BEAR, LABEL_BEAR]
bear_accuracy = n_bear_true_pos / n_bear_pred if n_bear_pred > 0 else 0

n_total_pred = n_bull_pred + n_bear_pred
n_total_true_pos = n_bull_true_pos + n_bear_true_pos
total_accuracy = n_total_true_pos / n_total_pred if n_total_pred > 0 else 0

pd.DataFrame([{"Prediction" : "Bull", "Accuracy" : bull_accuracy},
              {"Prediction" : "Bear", "Accuracy" : bear_accuracy},
              {"Prediction" : "Total", "Accuracy" : total_accuracy}
             ])


# In[ ]:




