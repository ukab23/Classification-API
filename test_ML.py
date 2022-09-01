from unicodedata import name
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
# from sklearn.externals import joblib
import joblib
# import mlflow.sklearn

class CreateModel():
     def __init__(self):
          self.df = pd.read_csv('train.csv')
          self.df_test = pd.read_csv('test.csv')
          self.include = ['Age', 'Sex', 'Pclass', 'Embarked', 'Survived']
          self.df = self.df[self.include]
          self.df_test = self.df[self.include]
     # print(df.head(3))

# Data Preprocessing
     def train(self):
          print("Creating models..")
          categoricals = []
          for col, col_type in self.df.dtypes.iteritems():
               if col_type == 'O':
                    categoricals.append(col)
               else:
                    self.df[col].fillna(0, inplace=True)

          categoricals_test = []
          for col, col_type in self.df_test.dtypes.iteritems():
               if col_type == 'O':
                    categoricals_test.append(col)
               else:
                    self.df_test[col].fillna(0, inplace=True)


          df_ohe = pd.get_dummies(self.df, columns=categoricals, dummy_na=True)
          dependent_variable = 'Survived'
          x_train = df_ohe[df_ohe.columns.difference([dependent_variable])]
          y_train = df_ohe[dependent_variable]

          df_ohe_test = pd.get_dummies(self.df_test, columns=categoricals_test, dummy_na=True)
          dependent_variable = 'Survived'
          x_test = df_ohe_test[df_ohe_test.columns.difference([dependent_variable])]
          y_test = df_ohe_test[dependent_variable]

          # mlflow.set_experiment("Classification_API")
          # mlflow.set_tracking_uri('http://localhost:8000')
          # mlflow.sklearn.autolog()
          # Logistic Regression classifier
          # mlflow.start_run("Logistic_Regression")
          lr = LogisticRegression()
          lr.fit(x_train, y_train)
          y_pred = lr.predict(x_test)
          print("LR accuracy",accuracy_score(y_test, y_pred))
          # mlflow.end_run("Logistic_Regression",status='FINISHED')

          # Support vector Machine classifier
          # mlflow.start_run("SVC")
          svc = SVC(gamma='auto')
          svc.fit(x_train, y_train)
          y_pred = svc.predict(x_test)
          print("SVM accuracy",accuracy_score(y_test, y_pred))
          # mlflow.end_run("SVC",status='FINISHED')

          # Naive Bayes classifier
          # mlflow.start_run("Naive_Bayes")
          nb = BernoulliNB()
          nb.fit(x_train, y_train)
          y_pred = nb.predict(x_test)
          print("NB accuracy",accuracy_score(y_test, y_pred))
          # mlflow.end_run("Naive_Bayes",status='FINISHED')

          # Save your model
          joblib.dump(lr, 'models\\lr_model.pkl')
          # print("LR Model dumped!")
          joblib.dump(svc, 'models\\svc_model.pkl')
          # print("SVC Model dumped!")
          joblib.dump(svc, 'models\\nb_model.pkl')
          # print("NB Model dumped!")

          # Load the model that you just saved
          # lr = joblib.load('models\\lr_model.pkl')
          # svc = joblib.load('models\\svc_model.pkl')
          # nb = joblib.load('models\\nb_model.pkl')

          # Saving the data columns from training
          model_columns = list(x_train.columns)
          joblib.dump(model_columns, 'models\\model_columns.pkl')
          print("Models have been saved..!")

if __name__ == '__main__':
     create = CreateModel()
     create.train()
