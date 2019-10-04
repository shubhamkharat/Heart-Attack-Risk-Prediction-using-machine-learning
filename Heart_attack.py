
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# read the file
df = pd.read_csv("sed.csv")
#print(data)
#print(data.head(5))
from sklearn.model_selection import train_test_split
X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
y = df.iloc[:, 13].values
#print(X)
split_test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state=10)
def train_model(X_train, y_train, X_test, y_test,tte, classifier, **kwargs):
    model = classifier(**kwargs)
    # train model
    model.fit(X_train, y_train)
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    print(model.predict(tte))

    return model
#input list
tte=[[67,1,4,160,286,0,2,108,1,1.5,2,3,3]]
# aquaracy check
model = train_model(X_train, y_train, X_test, y_test,tte, LogisticRegression)
model = train_model(X_train, y_train, X_test, y_test,tte, DecisionTreeClassifier)
#decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
s=clf.predict(tte)
risk=(s[0]/4)*100
print(risk)

