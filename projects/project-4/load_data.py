import numpy as np
import pandas as pd
import proj_lib
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore') 

####################################################### 
#  Main Program
####################################################### 
# this section loads data from file as raw data
# it performs data cleaning and some basic exploratory
# upon completion, a clean file will be saved

# local variables :
# indeed_df, datafile, clean_data

datafile = 'indeed_df.csv'
clean_data = 'indeed_clean.csv'
indeed_df = pd.read_csv(datafile)

print(indeed_df.info())

print("\nchecking for duplicated rows")
print(f"{indeed_df.duplicated().sum()} row(s) found")
indeed_df = indeed_df.drop_duplicates()
print("duplicaed row(s) removed.\n")

# indeed_df.groupby('Salary')['Salary'].count()[:5]
# indeed_df.groupby('Company')['Company'].count()[:5]

# only 762 records with salary range defined
cnt = indeed_df[indeed_df.Salary != 'None']['Salary'].count()
print(f"\n{(int(cnt * 1000 / indeed_df.shape[0])/10)}% of records have salary values")

# should use soup.decode_contents() in the first place
top5jobs = lambda x: [j for j in list(indeed_df.Title.unique()) if j.lower().find(x)>0][:5]
# List out those job related to Data Scientist
top5jobs('scientist')             # found &amp; should be removed
indeed_df['Title'] = indeed_df['Title'].apply(lambda x: x.replace('&amp;','&'))

####################################################### 
# Average annual salaries
indeed_df['AverageSalary'] = float(0)  #default value , to be used for prediction later on
indeed_df['Higher_Salary'] = 0  #default value since some rows has no salary range defined

####################################################### 
# Removal incorrect records
# Manual assign job category with some keywords, check the left overs
indeed_df['Category'] = 0  # default 0 as unassigned category
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('scien')] = 1
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('learn')] = 1
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('intelligen')] = 2
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('bi')] = 2
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('data warehous')] = 2
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('analy')] = 3
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('engine')] = 4
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('architect')] = 5
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('developer')] = 6
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('python')] = 6
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('consultant')] = 7
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('chief')] = 7
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('lecture')] = 8
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('data')] = 9
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('specialist')] = 10
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('research')] = 11
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('oracle')] = 12
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('contract')] = 13
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('intern')] = 14
df = indeed_df['Title'][indeed_df.Category==0]
cnt = df.count()
print(f'found {cnt} records with job title not relevant, to be removed')
indeed_df.drop(indeed_df['Title'][indeed_df.Category==0].index , inplace = True) 
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('contract')] = 13
indeed_df['Category'][indeed_df.Title.str.lower().str.contains('intern')] = 14

indeed_df['Category']=indeed_df['Category'].apply(lambda x: chr(64+x))

####################################################### 
# save the "clean" dataframe indeed_df into csv files
indeed_df.to_csv(clean_data, index=False, encoding='utf-8')
proj_lib.master_df = indeed_df
proj_lib.master_df.name = clean_data

print(f"File have been saved as {clean_data}")

