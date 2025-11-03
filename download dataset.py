import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle

df = pd.read_csv("language_identification_dataset.csv", encoding="utf-8")

"""
df_len = pd.DataFrame(data=(df['Text'].str.len()))
df_len['language'] = df.language
#len(dataset[dataset.language=='Estonian'])


lang_dic = {}
for lang in df.language.unique():
    curr_df = df[df.language==lang]
    lang_dic[lang] = curr_df
"""

# consider preprocessing empty space \n space tab etc.
#Chinese "。，“”:《》（）"
#Arabic "٪"
def preprocess(string):
    punctuations = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
    for i in range(len(string)):
        if string[i] in punctuations:
            string = string.replace(string[i], '_')
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
        print('-------------------------------------------')
        print(language)
        print(current_dic)
        all_dics[language] = current_dic
    #with open(f'{n}-gram.pkl', 'wb') as f:
        #pickle.dump(all_dics, f)
    return all_dics

a = export_ngram_pkl(2)

with open('2-gram.pkl', 'rb') as f:
    bigram = pickle.load(f)


