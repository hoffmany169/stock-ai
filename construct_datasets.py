#!/usr/bin/env python
# coding: utf-8

# In[5]:


import yfinance as yf
SYMBOL = "IFX.DE"
HISTORY = "10y"

all_day_k = yf.Ticker(SYMBOL).history(period=HISTORY, interval="1d")


# In[7]:


# remove meaningless columns
all_day_k = all_day_k.drop(columns=["Dividends", "Stock Splits"])

# remove latest row because it may be incomplete
all_day_k = all_day_k[:-1]


# In[15]:


## construct datasets
import numpy as np
import pandas as pd

PAST_WIN_LEN = 100
CLASSES = ["Bull", "Bear"]
LABEL_BULL = CLASSES.index("Bull")
LABEL_BEAR = CLASSES.index("Bear")

past_data, labels = [], []
today_i = PAST_WIN_LEN - 1
for today_i in range(len(all_day_k)):
    # get day-k in the past 100-day window and move forwards 1-day window
    day_k_past = all_day_k[:today_i + 1]
    day_k_forward = all_day_k[today_i + 1:]
    if len(day_k_past) < PAST_WIN_LEN or len(day_k_forward) < 1:
        continue
    day_k_past_win = day_k_past[-PAST_WIN_LEN:]
    day_k_forward_win = day_k_forward[:1]
    # find label
    today_price = day_k_past_win.iloc[-1]["Close"]
    tomorrow_price = day_k_forward_win.iloc[0]["Close"]
    label = LABEL_BULL if tomorrow_price > today_price else LABEL_BEAR
    # store: columns are taken as lists of array
    ##values = day_k_past_win.values # convert to array, deprecated
    values = day_k_past_win.to_numpy()
    past_data.append(values)
    labels.append(label)

x, y = np.array(past_data), np.array(labels)


# In[35]:


## split dataset to training validation and test datasets
TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT = 0.7, 0.2, 0.1
# take the last portion to be the test dataset
test_split_index = -round(len(x) * TEST_SPLIT)
# split x and y to test and other datasets
x_other, x_test = np.split(x, [test_split_index])
y_other, y_test = np.split(y, [test_split_index])
# shuffle the remaining portion and split into training and validation datasets
train_split_index = round(len(x) * TRAIN_SPLIT)
# get indexes array with the length same as x_other
indexes = np.arange(len(x_other))
# create random order in indexes
rng = np.random.default_rng()
rng.shuffle(indexes)
# split indexes array
train_indexes, valid_indexes = np.split(indexes, [train_split_index])
# split x_other array
x_train, x_valid = x_other[train_indexes], x_other[valid_indexes]
# split y_other array
y_train, y_valid = y_other[train_indexes], y_other[valid_indexes]


# In[55]:


# show label distribution
label_distribution = pd.DataFrame([{"DataSet" : "train",
                                    "Bull" : np.count_nonzero(y_train == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_train == LABEL_BEAR)},
                                   {"DataSet" : "valid",
                                    "Bull" : np.count_nonzero(y_valid == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_valid == LABEL_BEAR)},
                                   {"DataSet" : "test",
                                    "Bull" : np.count_nonzero(y_test == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_test == LABEL_BEAR)}])


# In[59]:


# balance labels of test data: let labels of bull and bear are same
x_test_bull = x_test[y_test == LABEL_BULL] ## all bull label in x_test
x_test_bear = x_test[y_test == LABEL_BEAR] ## all bear label in x_test

min_n_labels = min(len(x_test_bull), len(x_test_bear)) # get minimum nuber of labels of bull and bear
# 
x_test_bull = x_test_bull[np.random.choice(len(x_test_bull), min_n_labels, replace=False), :]

x_test_bear = x_test_bear[np.random.choice(len(x_test_bear), min_n_labels, replace=False), :]

x_test = np.vstack([x_test_bull, x_test_bear])
y_test = np.array([LABEL_BULL] * min_n_labels + [LABEL_BEAR] * min_n_labels)

pd.DataFrame([{"DataSet" : "test",
                "Bull" : np.count_nonzero(y_test == LABEL_BULL),
                "Bear" : np.count_nonzero(y_test == LABEL_BEAR)}])


# In[70]:


# the arrays are saved with the keyword names.
np.savez("datasets.npz", x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test)


# In[ ]:




