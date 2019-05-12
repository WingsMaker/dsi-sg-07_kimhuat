import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
import proj_lib
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree  
#from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import recall_score
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsRegressor

tgt = 'AverageSalary'
features = proj_lib.features

X = proj_lib.master_df[features]
ypred = proj_lib.salary_model.predict(X)
proj_lib.master_df[tgt]=ypred

salary_mean = np.mean(proj_lib.master_df.AverageSalary)
proj_lib.master_df['Higher_Salary'] = proj_lib.master_df[tgt].apply(lambda sal : 1 if sal > salary_mean else 0)

# Saving the end results into file
proj_lib.master_df.to_csv(proj_lib.master_df.name, index=False, encoding='utf-8')
print(f"\nFile have been saved as {proj_lib.master_df.name}")

print('average salary per job category')
print( proj_lib.master_df.groupby('Category').AverageSalary.mean() )

# Matrix Report not available due to ypred and y were continuous variables

#list_cols = ['Title','AverageSalary', 'Category','Company', 'Summary', 'City']
list_cols = ['Title','AverageSalary', 'Category','Company']
print(proj_lib.master_df[list_cols][proj_lib.master_df.Salary=='None'].sort_values('Category').groupby('Category').head(1))

