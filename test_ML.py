import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.externals import joblib


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
include = ['Age', 'Sex', 'Pclass', 'Embarked', 'Survived']
df = df[include]
df_test = df[include]
# print(df.head(3))

# Data Preprocessing
categoricals = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df[col].fillna(0, inplace=True)

categoricals_test = []
for col, col_type in df_test.dtypes.iteritems():
     if col_type == 'O':
          categoricals_test.append(col)
     else:
          df_test[col].fillna(0, inplace=True)


df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)
dependent_variable = 'Survived'
x_train = df_ohe[df_ohe.columns.difference([dependent_variable])]
y_train = df_ohe[dependent_variable]

df_ohe_test = pd.get_dummies(df_test, columns=categoricals_test, dummy_na=True)
dependent_variable = 'Survived'
x_test = df_ohe_test[df_ohe_test.columns.difference([dependent_variable])]
y_test = df_ohe_test[dependent_variable]


# Logistic Regression classifier
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("LR accuracy",accuracy_score(y_test, y_pred))

# Support vector Machine classifier
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("SVM accuracy",accuracy_score(y_test, y_pred))

# Naive Bayes classifier
nb = BernoulliNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
print("NB accuracy",accuracy_score(y_test, y_pred))

# Save your model
joblib.dump(lr, 'lr_model.pkl')
# print("LR Model dumped!")
joblib.dump(svc, 'svc_model.pkl')
# print("SVC Model dumped!")
joblib.dump(svc, 'nb_model.pkl')
# print("NB Model dumped!")

# Load the model that you just saved
lr = joblib.load('lr_model.pkl')
svc = joblib.load('svc_model.pkl')
nb = joblib.load('nb_model.pkl')

# Saving the data columns from training
model_columns = list(x_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
