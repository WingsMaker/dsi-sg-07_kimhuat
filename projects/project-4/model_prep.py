import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore') 

import proj_lib

global indeed_df

class models_matrix(object):
    def __init__(self):
        self.model_list  = []
        self.model_score = []
        self.model_acc   = []
        self.model_rmse  = []
        self.model_rss   = []        
        pass
    
    def add(self, RegModel):
        self.model_list.append(RegModel.name)
        self.model_score.append(RegModel.meanscore)
        self.model_acc.append(RegModel.modelscore)
        self.model_rmse.append(RegModel.rmse)
        self.model_rss.append(RegModel.rss)

    def report(self):
        df = pd.DataFrame({'Model': self.model_list, 
                                   'mean score': self.model_score, 
                                   'accurary': self.model_acc, 
                                   'RMSE': self.model_rmse, 
                                   'RSS': self.model_rss                       
                                  }).sort_values("accurary", ascending=False)
        return df

####################################################### 
#  Main Program
####################################################### 
# This section defines the target and features
# and preparing for train and test data based on known salary data.
# StandardScaler is used to standardize them
#
# input files  :  indeed_salaries.csv

print('preparing data for regularization')

train_size = 600
salary_data = proj_lib.master_df[proj_lib.master_df.Salary != 'None'].copy()
salary_data.AverageSalary = salary_data.AverageSalary.apply(lambda x: float(x))

tgt = 'AverageSalary'
list_cols = list(proj_lib.master_df.columns)
list_cols.remove('Title')
list_cols.remove('Company')
list_cols.remove('Summary')
list_cols.remove('City')
list_cols.remove('Salary')
list_cols.remove('AverageSalary')
list_cols.remove('Category')
proj_lib.features = features = list_cols

print('list of features for regularization process:\n', list_cols)

proj_lib.trainX = X = salary_data[list_cols][:train_size]
proj_lib.trainY = y = salary_data[tgt][:train_size]

xTrain, xTest, yTrain, yTest = \
            train_test_split(X, y, random_state=78)

# standardize them
std = StandardScaler()
X_train_std = std.fit_transform(xTrain)
X_test_std = std.transform(xTest)

print('data preparation done, ready for regularization')

####################################################### 
# Regularization
####################################################### 
model_list  = []
model_score = []
model_acc   = []
model_rmse  = []
model_rss   = []

Reg_Matrix = models_matrix()
proj_lib.RegRidge = proj_lib.RegressorRidge(list_cols, xTrain, X_train_std, xTest, X_test_std, yTrain, yTest)
Reg_Matrix.add(proj_lib.RegRidge)

proj_lib.RegLasso = proj_lib.RegressorLasso(list_cols, xTrain, X_train_std, xTest, X_test_std, yTrain, yTest)
Reg_Matrix.add(proj_lib.RegLasso)

proj_lib.RegElasticNet = proj_lib.RegressorElasticNet(list_cols, xTrain, X_train_std, xTest, X_test_std, yTrain, yTest)
Reg_Matrix.add(proj_lib.RegElasticNet)

models_performance = Reg_Matrix.report()
print(models_performance)

rmse = np.min(models_performance.RMSE)
row = models_performance[models_performance.RMSE <= rmse].index[0]
model_select = ['Ridge','Lasso','ElasticNet'][row]
print('\nbest model with lowest RMSE is ', model_select )

print('regularization completed, next thing features elimination test\n')

####################################################### 
# Features Elimination 
####################################################### 

print(f"\nLasso eliminated some, left with: {proj_lib.RegLasso.columns.shape[0]} columns")
print( proj_lib.RegLasso.columns )

print(f"\nRidge eliminated some, left with: {proj_lib.RegRidge.columns.shape[0]} columns")
print( proj_lib.RegRidge.columns )

print(f"\nElasticNet eliminated some, left with: {proj_lib.RegElasticNet.columns.shape[0]} columns")
print( proj_lib.RegElasticNet.columns )

print('\nFeatures elimination process completed\n')
