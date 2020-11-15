# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:42:53 2020

"""
#import library
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#read dataset
dataset = pd.read_csv("term-deposit-marketing-2020.csv")
#print(dataset.head())
#print(dataset.describe())
#print(dataset.info())

#Correlation (the numerical column)
corre= dataset.corr() 
#print(corre)
#print(dataset.select_dtypes(include=["int64", "float64"]).describe().T)

#preprocesssing

le= LabelEncoder()
dataset["job"] = le.fit_transform(dataset["job"])
dataset["marital"] = le.fit_transform(dataset["marital"])
dataset['education'] = le.fit_transform(dataset["education"])
dataset['default'] = le.fit_transform(dataset["default"])
dataset['housing'] = le.fit_transform(dataset["housing"])
dataset['loan'] = le.fit_transform(dataset["loan"])
dataset['contact'] = le.fit_transform(dataset["contact"])
dataset['month'] = le.fit_transform(dataset["month"])
dataset['y'] = le.fit_transform(dataset["y"])


# Use min/max feature scaling to normalize the dataset
minmaxscaler = preprocessing.MinMaxScaler(feature_range = (0,1))
dataset = minmaxscaler.fit_transform(dataset)
dataset = pd.DataFrame(dataset, columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                                           'housing', 'loan', 'contact', 'day', 'month', 'duration', 
                                           'campaign', 'y'])
print(dataset.head())


data=['age', 'job', 'marital', 'education', 'default', 'balance', 
      'housing','loan', 'contact', 'day', 'month', 'duration', 'campaign']
x=dataset[data]
y=dataset.y


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

logreg_predict = logreg.predict(x_test)
print('Logistic regression accuracy score= ',logreg.score(x_test, y_test) )

logreg_p= cross_val_score(logreg, x, y, cv= 5) 
print('The accuracy score by having 5 folds=', logreg_p)
print('The mean of all the accuracy= ',sum(logreg_p)/len(logreg_p))

print(confusion_matrix(y_test, logreg_predict))
print(classification_report(y_test, logreg_predict))


#Decision Tree
dec_tree=DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)

dec_tree_predict= dec_tree.predict(x_test)
print('Decision tree accuracy score= ', dec_tree.score(x_test, y_test))

dec_tree_p = cross_val_score(dec_tree, x, y, cv= 5)
print('The accuracy score by having 5 folds=',dec_tree_p)
print('The mean of all the accuracy= ',sum(dec_tree_p)/len(dec_tree_p))


print(confusion_matrix(y_test,dec_tree_predict))
print(classification_report(y_test, dec_tree_predict))

#Random Forest 

ran_for= RandomForestClassifier()
ran_for.fit(x_train, y_train)

ran_for_predict= ran_for.predict(x_test)
print('Random forest classifier accuracy score= ',ran_for.score(x_test, y_test))

ran_for_p = cross_val_score(ran_for, x, y, cv= 5)
print('The accuracy score by having 5 folds= ',ran_for_p)
print('The mean of all the accuracy=',sum(ran_for_p)/len(ran_for_p))

print(confusion_matrix(y_test, ran_for_predict))
print(classification_report(y_test, ran_for_predict))


















