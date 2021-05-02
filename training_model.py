#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return "Unknown"
    
def shorter_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer','Don','the Countess', 'Dona','Lady','Sir']:
        return 'Royalty'
    elif title == 'Mme':
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title

df['Title'] = df.Name.map(lambda x: get_title(x))
df['Title'] = df.apply(shorter_titles, axis =1)

df.loc[df['Fare'] > 400, 'Fare'] = df['Fare'].median()
df.loc[df['Age'] > 70, 'Age'] = df['Age'].median()

df['Age'].fillna(df['Age'].median(), inplace = True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)


df.drop(["Name","Ticket","PassengerId", "Cabin"], axis =1, inplace = True)        #Delete columns

df.Sex.replace(('male','female'), (0,1), inplace = True)
df.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Royalty','Officer'), (0,1,2,3,4,5,6,7), inplace = True)

X = df.drop("Survived", axis =1)
y = df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

randomForest = RandomForestClassifier()
randomForest.fit(X_train, y_train)

pickle.dump(randomForest, open("titanic_RF_Model.sav", 'wb'))


# In[ ]:




