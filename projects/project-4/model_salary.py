import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import proj_lib, sys

####################################################### 
#  Main Program
####################################################### 
# This section will automate the most tedious part of ML
# by intelligently exploring thousands of possible pipelines 
# to find the best one for indeed.com jobs data.
#
# input file  :  indeed_clean.csv
# output : optimized model using AutoML(TPOT)


clean_data = 'indeed_clean.csv'
indeed_df = pd.read_csv(clean_data)
train_size = 600
salary_data = indeed_df[indeed_df.Salary != 'None'].copy()
salary_data.AverageSalary = salary_data.AverageSalary.apply(lambda x: float(x))

tgt = 'AverageSalary'
list_cols = proj_lib.features

X = proj_lib.trainX 
y = proj_lib.trainY

proj_lib.salary_model = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=11, min_samples_split=9)),
    RandomForestRegressor(bootstrap=True, max_features=0.6500000000000001, min_samples_leaf=7, min_samples_split=14, n_estimators=100)
)

for n_times in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                     train_size=0.75, test_size=0.25,  random_state=(n_times+1964))
    proj_lib.salary_model.fit(X_train, y_train)
    results = proj_lib.salary_model.predict(X_test) 

salary_model_mae = mean_absolute_error(results, y_test)
salary_model_mscore = proj_lib.salary_model.score(X_test, y_test)
salary_model_mse = mean_squared_error(y_test, results)
salary_model_rmse = np.sqrt(salary_model_mse)
salary_model_rscore = r2_score(y_test, results)

print('\nSummary')
print('mse        : %.4f' % salary_model_mse)
print('rmse       : %.4f' % salary_model_rmse)
print('mean score : %.4f' % salary_model_mscore)
print('mae        : %.4f' % salary_model_mae)
print('r2 score   : %.4f' % salary_model_rscore)

