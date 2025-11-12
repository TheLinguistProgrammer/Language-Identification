import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# consider preprocessing empty space \n space tab etc.
#Arabic "٪"
def preprocess(string):
    punctuations = """ \n\t!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。，“”:《》（）٪"""
    for i in range(len(string)):
        if string[i] in punctuations:
            string = string.replace(string[i], '_')
    string = string.replace('_', '')
    return string

def string2ngram(string, n):
    ngram_array = []
    for i in range(len(string)):
        if i <  len(string) - (n-1):
            current_ngram = []
            for j in range(n):
                current_ngram.append(string[i+j])
            ngram_array.append(current_ngram)
    return ngram_array

def ngram2dic(array:list):
    dic = {}
    for item in array:
        key = ''.join(item)
        if key in dic:
            dic[key] += 1
        else:
            dic[key] = 1
    # sorting dic
    sorted_dic = dict(sorted(dic.items(), key=lambda x:x[1], reverse=True))
    return sorted_dic

def export_ngram_pkl(n):
    all_dics = {}
    for language in df.language.unique():
        current_dic = {}
        current_dic = Counter(current_dic)
        new_df = df[df.language==language].Text.apply(lambda string: ngram2dic(string2ngram(preprocess(string), n)))
        for i in range(len(new_df) - 1):
            current_dic += Counter(new_df.iloc[i])
        all_dics[language] = current_dic
    with open(f'{n}-gram.pkl', 'wb') as f:
        pickle.dump(all_dics, f)
    return all_dics

def trim_ngram_dic(ngram, num):
    """return the n-most common n-grams
    trim_ngram(ngram, 5) returns the 5 most frequent n-grams
    ngram must be a dictionary of dictionaries
    """
    counter = num
    current_lang_dic = {}
    for key, value in ngram.items():
        if(counter>0):
            current_lang_dic[key] = value
            counter -= 1
    return current_lang_dic

def trim_ngram_array(ngram, num):
    """returns the most common ngrams without their frequency"""
    counter = num
    ngram_list = []
    for key, value in ngram.items():
        if(counter>0):
            ngram_list.append(key)
            counter -= 1
    return np.array(ngram_list)
#a = export_ngram_pkl(4)

"""
with open('2-gram.pkl', 'rb') as f:
    bigram = pickle.load(f)
"""







