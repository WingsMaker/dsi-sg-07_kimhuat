import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import proj_lib, sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score

import warnings
warnings.filterwarnings('ignore') 

####################################################### 
#  Main Program
####################################################### 
# This section does the valdation using holdout data

clean_data = 'indeed_clean.csv'
indeed_df = pd.read_csv(clean_data)
train_size = 600
salary_data = indeed_df[indeed_df.Salary != 'None'].copy()
salary_data.AverageSalary = salary_data.AverageSalary.apply(lambda x: float(x))

tgt = 'AverageSalary'
list_cols = proj_lib.features

proj_lib.holdoutX = X = salary_data[list_cols][train_size:]
proj_lib.holdoutY = y = salary_data[tgt][train_size:]

print('Model Lasso :')
selector = proj_lib.RegLasso.selector
ypred = selector.predict(X)
mse = mean_squared_error(y, ypred)
print('mse        : %.4f' % mse)
print('rmse       : %.4f' % np.sqrt(mse))
print('mean score : %.4f' % selector.score(X, y))
print('mae        : %.4f' % mean_absolute_error(ypred, y))
print('r2 score   : %.4f' % r2_score(y, ypred))

print('\nModel Pipeline :')    
ypred = proj_lib.salary_model.predict(X)
mse = mean_squared_error(y, ypred)
print('mse        : %.4f' % mse)
print('rmse       : %.4f' % np.sqrt(mse))
print('mean score : %.4f' % proj_lib.salary_model.score(X, y))
print('mae        : %.4f' % mean_absolute_error(ypred, y))
print('r2 score   : %.4f' % r2_score(y, ypred))
