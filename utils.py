import numpy as np
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

def convert_time(time_):
    if time_ < 60:
        return time_, 'sec(s)'
    elif 60 < time_ < 3600:
        return (time_ / 60), 'min(s)'
    elif time_ > 3600:
        return (time_ / 3600), 'hour(s)'

def transformation(C):

    # Transform the 0-1 challenge to -1 and +1.
    V = 2 * C - 1
    V = np.fliplr(V)

    # Compute the cumulative product (right to left)
    V = np.cumprod(V, axis=1, dtype=np.int8)
    V = np.fliplr(V)

    return V

# responses_to_labels
# Converts a 1 or -1 to a one-hot encoding of the potential outputs

def responses_to_labels(r):
    l = np.zeros((r.shape[0], 2))
    for i in range(0, r.shape[0]):
        if r[i] == 1:
            l[i] = [0, 1]
        else:
            l[i] = [1, 0]
    
    l = np.asarray(l, dtype = np.int8)
    return l

# labels_to_responses
# Converts the output of a softmax function to a 1 or -1

def labels_to_responses(l):
    r = []
    for i in range(0, l.shape[0]):
        if l[i][0] < l[i][1]:
            r.append(1)
        else:
            r.append(-1)

    r = np.asarray(r, dtype = np.int8)
    return r

'''
Get a list of keys from dictionary which has the given value
'''
def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys
