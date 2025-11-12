import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("language_identification_dataset.csv", encoding="utf-8")
np.mean(pd.DataFrame(data=(df['Text'].str.len()))) #356.0332272727273
df.iloc[0].Text


def string2ord(string:str, length):
    """
    removed if long,
    padded if short
    """
    ord_list = []
    for character in string:
        ord_list.append(ord(character))  
    pad_ord_list = ord_list[:length]
    for i in range(length - len(pad_ord_list)):
        pad_ord_list.append(0)
    return pad_ord_list
    #return np.array(pad_ord_list)

df.insert(2, 'vector', df.Text.apply(lambda text: string2ord(text, 300)))
data_classes = list(df.language.unique())
df.insert(3, 'labels', df['language'].apply(data_classes.index))

# NEURAL NETWORK PART
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
X = np.array(list(df.vector)) # changed the dimensions to (22000,300)
y = np.array(df.labels)
scaler1 = MinMaxScaler()
X = scaler1.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# y_train = np.array(y_train).reshape(-1,1)

nn_classifier = MLPClassifier(max_iter=100,
                        tol=0.0000100,
                        activation='logistic',
                        solver='lbfgs',
                        learning_rate='constant',
                        learning_rate_init=0.05,
                        batch_size=32,
                        hidden_layer_sizes=(100,100))
nn_classifier.fit(X_train, y_train)

#verbose=True,

from sklearn.metrics import accuracy_score, confusion_matrix
predictions = nn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
cm = confusion_matrix(y_test, predictions)
#print(cm)

# VISUALIZATION
from yellowbrick.classifier import ConfusionMatrix
confusion_matrix = ConfusionMatrix(nn_classifier, classes=df.language.unique())
confusion_matrix.fit(X_train, y_train)
confusion_matrix.score(X_test, y_test)
confusion_matrix.show()


