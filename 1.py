# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:33:29 2024

@author: Admin
"""

import pandas as pd
df=pd.read_csv()
df.head()
df.isnull().sum()
df.describe()
df.Outcome.value_counts()

#0 500
#1 268
#There is slight imbalance in out dataset but since it is not 
#Train test split
x=df.drop("Outcome",axis="columns")
y=df.Outcome
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_scaled[:3]
#In order to make your data balanced while splitting you can
#use stratify
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled, y,stratify=y,random_state=10)
x_train.shape
x_test.shape
y_test.value_counts()

#Train using standalone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#Here k fold cross validation is used
scores=cross_val_score(DecisionTreeClassifier, x,y,cv=5)
scores
scores.mean()


from sklearn.ensemble import BaggingClassifier

bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0)

bag_model.fit(x_train,y_train)
bag_model.oob_score_



bag_model.score(x_test,y_test)


bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0)

scores=cross_val_score(bag_model,x,y,cv=5)
scores
scores.mean()


from sklearn.ensemble import RandomForestClassifier
scores=cross_val_score(RandomForestClassifier(n_estimators=50),x,y,cv=5)
scores.mean()


