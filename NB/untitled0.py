
# NAIVE BAYES
import numpy as np
import pandas as pd
import seaborn as sns
from math import log
df = pd.read_csv("language_identification_dataset.csv", encoding='UTF-8')



def string2token(string):
    """ check if the language is space-based"""
    token_length_limit = 12
    space_count = string.count(' ')
    current_string_length = len(string)
    expected_spaces = current_string_length / token_length_limit
    if space_count < expected_spaces:
        #print('JAPANESE OR CHINESE')
        #print(string)
        ''
    else:
        print('\n', '==========SPACE LANGUAGE===========')
        print(string)


language_dictionary = {}
for language in df.language.unique():
    language_dictionary[language] = df[df.language==language]


#for sentence in language_dictionary['Japanese'].Text:
    #string2token(sentence)

"""
Some sentences may contain one or two spaces,
which can prevent correct categorization for preprocessing.

Thus a better idea is to come up with a measure ratio of letters to spaces

This data is horrible. there are many instances where languages are mixed together
"""



""" GOAL: FIND THE OUTLIERS IN THE DATASET USING THREE VARIABLES:
    1. MIN
    2. MAX
    3. AVERAGE """
# CONVERT TO ORD
def string2ord(string:str):
    """ Converts a string to an array of values using the default ord() function"""
    ord_list = []
    for character in string:
        ord_list.append(ord(character))  
    return ord_list

def string2ord_padded(string:str, length):
    """ Converts a string to an array and pads and shortens based on length"""
    ord_list = []
    for character in string:
        ord_list.append(ord(character))
    ord_list = ord_list[:length]
    for i in range(length - len(ord_list)):
        ord_list.append(0)
    return ord_list

# TODO: Standard Deviation may be a good measure
def mean(array:list):
    return sum(array)/len(array)

# TODO: NOTE THAT THIS IS WITHOUT PREPROCESSING, SO THE COMMON FACTORS IN ALL LANGUAGES
# ARE NOT REMOVED WHICH WILL PROBABLY MAKE THE VALUES LESS RELIABLE (NUMBERS MAY BE USELESS FOR EXAMPLE)


#df['ord'] = df.Text.apply(lambda text: string2ord(text)) #non-trimmed/padded
df['ord'] = df.Text.apply(lambda text: string2ord_padded(text, 120)) #trimmed/padded
df['mean'] = df.ord.apply(lambda array: log(mean(array))) # can try with or without log
df['max'] = df.ord.apply(lambda array: max(array))

"""
sns.set_theme(rc={'figure.figsize':(20,5)})
sns.boxplot(x='language', y='mean', data=df)
"""

# FIND OUTLIERS USING UNIGRAMS?
# the idea is that the some characters, such as space, should occur much
# less in language such as chinese and japanese and if they occur with high
# frequency, then it means that it's an outlier
# could also apply to arabic characters
def ord_frequency(sub_df):
    """
    

    Parameters
    ----------
    sub_df : TYPE
        Example: df[df.language=='Swedish']

    Returns
    -------
    None.

    """   
    all_ords = {}
    for array in sub_df.ord:
        for num in array:
            if num in all_ords:
                all_ords[num] += 1
            else:
                all_ords[num] = 1
    all_ords = dict(sorted(all_ords.items(), key=lambda x:x[1], reverse=True))
    return all_ords

english_frequencies = ord_frequency(df[df.language=='English'])
import matplotlib.pyplot as plt
#plt.plot(list(english_frequencies.values()))


#sns.barplot(x=list(english_frequencies.keys())[:50], y=list(english_frequencies.values())[:50])

# CONVERT TO LOG
#log_of_array = lambda array: [log(num) for num in array]
#log_of_array(english_frequencies.values())
#sns.barplot(x=list(english_frequencies.keys())[:50], y=list(log_of_array(english_frequencies.values()))[:50])
x = list(map((lambda num: chr(num)), english_frequencies.keys()))

#sns.barplot(x=x[:], y=list(english_frequencies.values())[:])


from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
clf = CategoricalNB() # unable to handle unseen data
#clf = GaussianNB() # performs poorly 0.42
#clf = BernoulliNB() # TERRIBLE PERFORMANCE?!? 0.05!
#X = np.array(df.ord)
X = np.array(list(df.ord))
y = np.array(df.language)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
# clf.predict(X_test)
score = clf.score(X_test, y_test)

"""one big issue with this approach (CategoricalNB) is the following:
it seems like if the number (training data thus ord) is not included in
the training data, the classifier will give an error

== Error message ==
index 65354 is out of bounds for axis 1 with size 65310"""



# one solution may be to normalize the input data
