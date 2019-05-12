import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import requests
import bs4
from bs4 import BeautifulSoup
from random import randint
from time import sleep
import sys, os, time, re
from os.path import expanduser
from subprocess import check_output
import pickle
import warnings
warnings.filterwarnings('ignore') 

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

from sklearn.pipeline import Pipeline

global RegLasso, RegRidge, RegElasticNet, features, salary_model
global trainX, trainY, holdoutX, holdoutY , master_df

def delayshortwhile():
    sleep(randint(3,10))
    return

def soupfile(pgnum):
    return f"./indeed/indeed_pg{pgnum}.txt"  

def save_soup(soup, pgnum):
    fn = soupfile(pgnum)
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())    
    f.close()
    return

def FindLastPage(urlref):
    ## Find the last page of the search results
    url = urlref.format(1000)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml', from_encoding='utf-8')
    lastpage = soup.find(name = 'div', attrs = {'class':'pagination'}).find_all('a')[-1]['href'].split('=')[2]
    max_page = int(lastpage) + 10
    lastpage = urlref.format(max_page)
    return max_page, lastpage

def LoadWebPages(driver, query_url, num_from, num_to):
    max = num_to + 1
    for n in range(num_from, max, 10):
        pagelink = query_url.format(n)
        # print(pagelink)
        driver.get(pagelink)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')        
        save_soup(soup, n)
        if n % 50 ==0:
            fn = soupfile(n)
            print("saving soupfile ",fn)                
        delayshortwhile()
    print("all pages downloaded.")
    return

def showrecord(title, company, location, salary, summary):
    print("Title   : ", title)
    print("Company : ", company)
    print("City    : ", location)
    print("Salary~=: ", salary)
    print("Summary : ",summary)
    return

def process_webpage(df, fn, Verify = False):
    try:
        with open(fn, 'r', encoding='utf-8') as f:
           soup = BeautifulSoup(f, "html.parser")
        f.close()    
    except:
        return False
    for j in soup.find_all(name = 'div', attrs = {'class':'jobsearch-SerpJobCard unifiedRow row result clickcard'}):
        jc=list(j.children)
        salary = j.find(name = 'div', attrs = {'class':'salarySnippet'})
        salary = 'None' if salary==None else salary.text.replace("\n"+""*11,"").strip()
        title=jc[1].a['title']
        company = jc[3].span.get_text().strip().replace("\n","").replace('  ','')
        location = jc[3].find(name = 'div', attrs = {'class':'recJobLoc'})['data-rc-loc'].strip()
        summary = j.find(name = 'div', attrs = {'class':'summary'}).text.replace("\n","")
        summary = " ".join([word for word in summary.split(" ") if word !=""])
        if Verify:
            if len([eval(var) for var in ['title', 'company', 'location'] if var=='']) > 0:
                print('Following files containing incomplete record :')
                showrecord(title, company, location, salary, summary)
                return False
            else:
                continue
        else:
            if dfinsert(df, title, company, location, salary, summary) == False:
                print("unable to save record into dataframe for below record")
                showrecord(title, company, location, salary, summary)
                return False
    return True

def dfinsert(df, title, company, location, salary, summary):
    rec = [title, company, location, salary, summary]
    if len(df)==0:
        df.loc[0] = rec
    try:
        r = df[(df.City == location) & (df.Title == title) & (df.Company == company)]['Title'].shape[0]
    except:
        print("Error D01")
        return False
    n = df.shape[0]
    if r == 0:        
        try:
            df.loc[n] = rec
        except:
            print("error ####, rec =", n)
            return False
    return True

def AddCatCol(df, prefix, subject, topic):
    colname = prefix + topic.replace(' ','').replace(',','').replace('.','')
    df[colname] = df[subject].apply( lambda x: 1 if topic in x else 0)
    return 


def best_alpha(CrossValidator, alpha_range, n_alphas, max_iter, Xtr, ytr, Xtt, ytt):
    # create an array of alpha values and select the best one with RidgeCV
    if max_iter == 0:
        if n_alphas  == 0:
            cv = CrossValidator(alphas = alpha_range, fit_intercept = True)
        else:
            cv = CrossValidator(n_alphas = n_alphas, fit_intercept = True)
    else:
        cv = CrossValidator(l1_ratio = alpha_range, 
                            n_alphas = n_alphas, 
                            cv = 10, max_iter=max_iter)
    cv.fit(Xtr, ytr)
    y_pred = cv.predict(Xtt)
    coefs = cv.coef_
    optimal_alpha = cv.alpha_
    return coefs, optimal_alpha

def model_run(optimal_cv, xTrain, X_train_std, X_test_std, yTrain, yTest):
        cv_scores = cross_val_score(optimal_cv, X_train_std, yTrain, cv=10)
        meanscore = np.mean(cv_scores)
        optimal_cv.fit(X_train_std, yTrain)
        modelscore = optimal_cv.score(X_test_std, yTest)    
        y_pred = optimal_cv.predict(X_test_std)
        rmse = np.sqrt(metrics.mean_squared_error(yTest, y_pred))
        rss = np.sum((yTest - y_pred) ** 2)
        rfecv = RFECV(estimator = optimal_cv, n_jobs = -1,  \
                      step=1, scoring = 'neg_mean_squared_error' ,cv=5)
        selector = rfecv.fit(xTrain, yTrain)
        return selector, meanscore, modelscore, rmse, rss

class RegressorLasso(object):
    def __init__(self):
        self.name = ''
        self.alpha = 0
        self.coefs = []
        self.meanscore = 0
        self.modelscore = 0
        self.rmse = 0
        self.rss = 0
        self.columns = []
        self.selector = object
        pass

    def __init__(self, list_cols, xTrain, X_train_std, xTest, X_test_std, yTrain, yTest):
        self.name = 'Lasso'
        self.coefs, self.alpha = best_alpha(LassoCV, [], 10, 0, \
                                            X_train_std, yTrain, X_test_std, yTest)
        optimal_cv = Lasso(alpha = self.alpha)
        self.selector, self.meanscore, self.modelscore, self.rmse, self.rss = \
            model_run(optimal_cv, xTrain, X_train_std, X_test_std, yTrain, yTest)
        self.columns = np.array(list_cols)[self.selector.support_]
        return
    
    def predit(self, X):
        return self.selector.predict(X)
    
class RegressorRidge(object):
    def __init__(self):
        self.name = ''
        self.alpha = 0
        self.coefs = []
        self.meanscore = 0
        self.modelscore = 0
        self.rmse = 0
        self.rss = 0
        self.columns = []
        self.selector = object
        pass

    def __init__(self, list_cols, xTrain, X_train_std, xTest, X_test_std, yTrain, yTest):
        self.name = 'Ridge'
        alphas = 10 ** np.arange(1, 5)
        ridge_weight = []
        for alpha in alphas:    
            ridge = Ridge(alpha = alpha, fit_intercept = True)
            ridge.fit(X_train_std, yTrain)
            ridge_weight.append(ridge.coef_)
        alpha_range = 10. ** np.arange(-2, 3)
        self.coefs, self.alpha =  best_alpha(RidgeCV,alpha_range,0,0,  \
                                             X_train_std, yTrain, X_test_std, yTest)
        optimal_cv = Ridge(alpha = self.alpha)
        self.selector, self.meanscore, self.modelscore, self.rmse, self.rss = \
            model_run(optimal_cv, xTrain, X_train_std, X_test_std, yTrain, yTest)
        self.columns = np.array(list_cols)[self.selector.support_]
        return
    
    def predit(self, X):
        return self.selector.predict(X)
    
class RegressorElasticNet(object):
    def __init__(self):
        self.name = ''
        self.alpha = 0
        self.coefs = []
        self.meanscore = 0
        self.modelscore = 0
        self.rmse = 0
        self.rss = 0
        self.columns = []
        self.selector = object
        pass

    def __init__(self, list_cols, xTrain, X_train_std, xTest, X_test_std, yTrain, yTest):
        self.name = 'ElasticNet'
        l1_ratios = np.linspace(0.01, 1.0, 25)
        self.coefs, self.alpha = best_alpha(ElasticNetCV, l1_ratios, 30, 10000, \
                                            X_train_std, yTrain, X_test_std, yTest)
        optimal_cv = ElasticNet(alpha = self.alpha)
        self.selector, self.meanscore, self.modelscore, self.rmse, self.rss = \
            model_run(optimal_cv, xTrain, X_train_std, X_test_std, yTrain, yTest)
        self.columns = np.array(list_cols)[self.selector.support_]
        return
    
    def predit(self, X):
        return self.selector.predict(X)
    
def save_obj(object, fn):
        try:
            outfile = open(fn,'wb')
            pickle.dump(object,outfile)
            #pickle.dump(object, open(outfile, 'wb'), protocol=2)
            #pickle.dump(object, open(outfile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            outfile.close()
        except:
            print('unable to save object into file ',fn);
        return 
def load_obj(object, fn):
        try:
            object = pickle.load(open(fn,'rb'))
            infile.close()
        except:
            print('unable to load object from file ',fn);
        return 

def jupyterlink(fn):
    # jupyterlink('proj_lib.py')
    netstat = check_output(["netstat", "-n"]).decode("utf8").replace('\r','').split('\n')
    ports=[ln for ln in netstat if ln.find('::1')>0]
    port=(ports[0].split('['))[1].split(']')[1].replace(':','').rstrip()
    prefix = 'http://localhost:' + port + '/edit'    
    jpath=os.getcwd().replace(os.path.expanduser('~'),prefix)
    jpath = jpath.replace('\\','/') + '/' 
    jpath = jpath + fn
    txt = 'Click below to view/edit the file:'
    print(f'{txt}:\n{jpath}')
    return 

def loadjobclass():
    ####################################################### 
    # Define dictionary
    JobClass = dict()
    JobClass['A']='Data Scientist, Machine Learning'
    JobClass['B']='BI, Business Intelligent, Data Warehousing'
    JobClass['C']='Analyst'
    JobClass['D']='Engineer'
    JobClass['E']='Solution Architect'
    JobClass['F']='Python Developer'
    JobClass['G']='Consultant, Manager, Chief'
    JobClass['H']='Lecturer'
    JobClass['I']='Data'
    JobClass['J']='Specialist'
    JobClass['K']='Research'
    JobClass['L']='Oracle'
    JobClass['M']='Contract'
    JobClass['N']='Intern'
    return JobClass

