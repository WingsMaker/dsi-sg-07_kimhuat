import numpy as np
import pandas as pd
#from numba import jit
import threading
import proj_lib
import warnings
warnings.filterwarnings('ignore') 

pd.set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None  

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFECV
from sklearn import metrics
import statsmodels.formula.api as sm

def features_table(rfc, X,  X_col, exact = True):
        wordlist = ['ai', 'analyst', 'analytics', 'anomaly', 'apps', 'architect', 'azure', 
        'bi', 'business', 'cloud', 'cognos', 'consultant', 'data', 'db2', 'deep',
        'learning', 'detection', 'engineer', 'etl', 'forecasting', 'hadoop', 
        'intelligence', 'machine', 'modeling', 'nlp', 'optimization', 
        'powerbi', 'python', 'python', 'qlik', 'r', 'sap', 'science', 'scientist', 
        'sql', 'tableau']
        df = proj_lib.master_df[proj_lib.master_df.Salary != 'None']
        X_median = np.median(df.AverageSalary.astype(float))
        #feature_importances = pd.DataFrame(rfc.feature_importances_, index = X.columns)
        #feature_importances.columns = ['feature', 'importance']
        #feature_importances.reset_index()
        feature_importances = pd.DataFrame(
            { 'feature':X.columns, 'importance':rfc.feature_importances_ })

        feature_medians = []
        feature_means = []
        for i in X.columns:            
            if exact:
                df1 = df[df[X_col] == i]['AverageSalary']
                i_median = np.median(df1.astype(float))
                i_mean = np.mean(df1.astype(float))
                feature_medians.append(i_median)
                feature_means.append(i_mean)                
            else:
                if i in wordlist:
                    df1 = df[df[X_col].str.lower().str.contains(i)]['AverageSalary']
                    i_median = np.median(df1.astype(float))
                    i_mean = np.mean(df1.astype(float))
                    feature_medians.append(i_median)
                    feature_means.append(i_mean)
                else:
                    feature_medians.append(0)
                    feature_means.append(0)

        feature_importances['median_value'] = feature_medians
        feature_importances['mean_value'] = feature_means
        feature_importances['over_or_under'] = [1 if i > X_median else 0 for i in feature_importances.median_value]
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        return feature_importances

# Use this to find the keywords to form the categorial columns
class ClassifierRandomForest(object):
    def __init__(self):
        self.accuracy = 0
        self.cvscore = 0
        self.feature_importances = pd.DataFrame
        pass

    def __init__(self, X_col):
        df = proj_lib.master_df[proj_lib.master_df.Salary != 'None']
        X = pd.get_dummies( df[X_col] )
        y = df['Higher_Salary']
        # 20% Test 80% Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)
        
        rfc = RandomForestClassifier(n_estimators=300, random_state=90)
        rfc.fit(X_train, y_train)

        rfc_pred = rfc.predict(X_test)
        self.accuracy = accuracy_score(y_test, rfc_pred)
        self.cvscore = cross_val_score(rfc, X, y, cv=10, n_jobs=-1)
        self.feature_importances = features_table(rfc, X, X_col, True)

class ClassifierWordVector(object):
    def __init__(self):
        self.accuracy = 0
        self.cvscore = 0
        self.feature_importances = pd.DataFrame
        pass

    def __init__(self, X_col):
        df = proj_lib.master_df[proj_lib.master_df.Salary != 'None']
        df.AverageSalary = df.AverageSalary.apply(float)
        X = df[X_col]
        y = df['Higher_Salary']
        cv = CountVectorizer(stop_words="english")
        cv.fit(X)
        X_trans = pd.DataFrame(cv.transform(X).todense(), 
                                     columns=cv.get_feature_names())
        
        X_train, X_test, y_train, y_test = train_test_split(
            np.asmatrix(X_trans), y, test_size=0.2, random_state=88, stratify=y)
        
        rfc = RandomForestClassifier(200, random_state=59)
        rfc.fit(X_train, y_train)

        rfc_pred = rfc.predict(X_test)
        self.accuracy = accuracy_score(y_test, rfc_pred)
        self.cvscore = cross_val_score(rfc, X_trans.as_matrix(), y.as_matrix(), cv=10, n_jobs=-1)
        self.feature_importances = rfc.feature_importances_
        print(rfc.feature_importances_)
        self.feature_importances = features_table(rfc, X_trans, X_col, False)


def ClassifyWords(ctype, fldcol, clfdict):
    colprefix = fldcol + '_'
    if ctype == 0:
        print(f'\nUsing Random Forest Classifier to explorer {fldcol}')
        clf = ClassifierRandomForest( fldcol )
    else:
        print(f'Using Count Vectorizer Classifier to explorer {fldcol}')
        clf = ClassifierWordVector( fldcol )
    
    print(f"Accuracy = {clf.accuracy}, features-counts for {fldcol}")    
    #print(clf.feature_importances)
    clfdict[ fldcol ] = list(clf.cvscore)
    print(f'Adding more feature columns for {fldcol}')
    for n in range(10):
        proj_lib.AddCatCol(proj_lib.master_df, colprefix, fldcol, clf.feature_importances.iloc[n].feature)
    return 
    

####################################################### 
#  Main Program
####################################################### 
# This section preparing the load some classifiers as model
# it load the processed datafile with average salary defined.
# Use it the run a few different model to find features columns
#
# Columns   Classifier Used:
# Company   Random Forest 
# City      Random Forest 
# Summary   Count Vectorizer
#
# input files  :  indeed_clean.csv 
# output files :  indeed_clean.csv 

indeed_df = proj_lib.master_df

clfdict = dict()
#clf=ClassifyWords(1, 'Summary', clfdict)

fld = ['Summary', 'Company', 'City']
clftype = [1, 0, 0]
threads = []
for idx in range(len(fld)):
    t = threading.Thread(target=ClassifyWords, args=(clftype[idx], fld[idx], clfdict))
    threads.append(t)
    
[t.start() for t in threads]
[t.join() for t in threads]

df = pd.DataFrame(clfdict)
print(df)

list_cols = list(proj_lib.master_df.columns)
n = list_cols.index('Higher_Salary')
features = list_cols[n:]
proj_lib.features = features

proj_lib.master_df.to_csv(proj_lib.master_df.name, index=False, encoding='utf-8')
print("updated datafile with extra columns")
