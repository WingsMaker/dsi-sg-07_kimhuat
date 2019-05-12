# Redo importing of modules if you skip the question 1 directly

import numpy as np
import pandas as pd
import proj_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Objective : Create a classification model to predict job class as :
jclass = proj_lib.loadjobclass()
print(jclass)

indeed_df = pd.read_csv('indeed_clean.csv')
proj_lib.master_df = indeed_df
proj_lib.master_df.name = 'indeed_clean.csv'
#proj_lib.master_df.info()

tgt = 'Category'
list_cols = [col for col in list(indeed_df.columns) if col != tgt] 
list_cols.remove('Salary')
list_cols.remove('Title')
list_cols.remove('Company')
list_cols.remove('Summary')
list_cols.remove('City')
list_cols.remove('AverageSalary')
X = indeed_df[list_cols]
y = indeed_df[tgt]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)

print(X.shape, X_train.shape, X_test.shape, y.shape, y_train.shape, y_test.shape)

# define the model, it uses SGD as the solver
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred.shape)

print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# random pick some results, unable to show
for n in range(10):    
    jt = n + 200
    jc = y_pred[jt] 
    jdesc = jclass[jc]
    print(f'job category #{jt} type={jc} - {jdesc}')