import string
import pandas as pd
import numpy as np
from multi_class_const import _word2vec_model, _vector_dim, customer_df
def delete_punctuation(s):
    """ return string with all the punctualtion marks replaced by spacebars"""
    for c in string.punctuation:
        s = s.replace(c, ' ')
    new_s = s
    while(True):
        new_s = s.replace('  ', ' ')
        if len(new_s) == len(s): break
        s = new_s
    return s

def get_phrase_vec(s):
    """ based on the model this returns real vector for a phrase as a sum of the vectors for all the words in the pharse"""
    s = delete_punctuation(s)
    l1 = s.split(' ')
    arr = np.array([0.0]*_vector_dim)
    for w in l1:
        try:
            arr1 = _word2vec_model[w]
            arr+=arr1
        except KeyError: continue
    return arr

def get_vec_array(arr):
    """ string array on input and vec array on output"""
    n = len(arr)
    vec_arr=np.zeros((n,_vector_dim))
    for i in range(n):
        if i%100==0: print(i)
        s = arr[i]
        vec_arr[i] = get_phrase_vec(s)
    return vec_arr
        
