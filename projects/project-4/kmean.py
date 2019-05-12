# Redo importing of modules if you skip the question 1 directly

import numpy as np
import pandas as pd
import proj_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

if type(jclass) != dict:
    jclass = proj_lib.loadjobclass()
job_cluster = len(jclass)
print(job_cluster)

k_mean = KMeans() #default k=8
k_mean.fit(X)

# Labels and centroids for 8 Clusters
labels = k_mean.labels_
print(labels)
clusters = k_mean.cluster_centers_
clusters[0]
print(silhouette_score(X, labels))

# 14 Clusters
k_mean3 = KMeans(n_clusters = job_cluster)
k_mean3.fit(X)
labels_3 = k_mean3.labels_
print(silhouette_score(X, labels_3))

# build the model with the found optimal parameters
k_mean_opt = KMeans(n_clusters = job_cluster)
k_mean_opt.fit(X)
labels_opt = k_mean_opt.labels_
print(len(labels_opt), X.shape, y.shape)

# random pick some results, unable to show
for n in range(10):    
    jt = n+200
    j = labels_opt[n+100] + 65
    jc = chr(j)    
    jdesc = jclass[jc]
    print(f'job category #{jt} type={jc} - {jdesc}')



