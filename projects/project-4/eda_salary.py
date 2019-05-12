import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import proj_lib, time

def avg_sal(sal):
    if sal=='None':
        return float(0)
    sal_key = ['an hour', 'a week', 'a month', 'a year'] 
    sal_rate = [2080, 52 , 12 , 1]
    sal_mean = float(0)
    for n in range(4):
        if sal.find(sal_key[n])>0:
            if sal.find(sal_key[n]) >0:
                sal_text = sal.replace('$','').replace(',','').replace(sal_key[n],'-')
                sal_text = sal_text.split('-')[:2]
                sal_text = [txt for txt in sal_text if txt!='']
                sal_low = float(sal_text[0])
                if len(sal_text) >= 2:
                    sal_high = float(sal_text[1])
                    sal_mean = np.mean([sal_high , sal_low ])
                else:
                    sal_mean = sal_low
                sal_mean = sal_mean * sal_rate[n]   # convert to annual
    return sal_mean

def highsal(sal):
    global salary_mean
    if sal=='None':
        return 'None'
    if sal > salary_mean:
        return 1
    return 0

def Top10_barplot(df, topic,  xfield = 'Title', yfield = 'AverageSalary'):
    maxrow = 10        
    df1 = df[[xfield, yfield]][df.Title.str.contains(topic)].groupby(xfield).mean()
    df1 = df1.sort_values(by=yfield, ascending=False)[:maxrow]
    print(df1)
    #ax = df1.plot(kind='barh', figsize=(6, 6), color='#86bf91', zorder=2, width=0.85)
    #for i, v in enumerate(df1[yfield]):
    #    vv = int(float(v/100))/10
    #    ax.text( v +2, i , str(vv) )
    #ax.set_ylabel(topic, labelpad=20, weight='bold', size=12)
    #ax.set_xlabel("Average Annual Salary (K)", labelpad=20, weight='bold', size=12)
    #plt.show()
    return 

####################################################### 
#  Main Program
####################################################### 
# this section performs EDA on Salary and Title columns
# For Salary, convert the rates like hourly, weekly, monthly, yearly.
# into annual salary rate
# convert the salary range into average annual rate
# Plot a graph to visualised the distribution

df = proj_lib.master_df[proj_lib.master_df.Salary != 'None']
df['Salary'] =df['Salary'].apply(avg_sal)
proj_lib.master_df['AverageSalary'] = proj_lib.master_df['Salary'].apply(avg_sal)
proj_lib.master_df['AverageSalary'] = proj_lib.master_df['AverageSalary'].astype(float)

# Statistical Information
salary_low = np.min(df.Salary)
print('minimum :' ,salary_low)
salary_high = np.max(df.Salary)
print('maximum :' ,salary_high)
salary_median = np.median(df.Salary)
print('median :' ,salary_median)
salary_mean = np.mean(df.Salary)
print('mean :' ,salary_mean)
std = np.std(df.Salary)

# plotting salary distribution
# with vertical lines to represent the mean and median salary
qt_left = salary_mean - std
qt_right = salary_mean + std
ax = sns.distplot(df.Salary)
ymin, ymax = ax.get_ybound()
ym = float(int(ymax + 1)/2)

ax.axvline(salary_median, lw=2.5, ls='dashed', color='black')
ax.axvline(salary_mean, lw=2.5, ls='dashed', color='red')
ax.axvline(qt_left, lw=2.5, ls='dashed', color='orange')
ax.axvline(qt_right, lw=2.5, ls='dashed', color='orange')

plt.title('Salary Distribution\nMedian=black, Mean=red, StdDev=Orange')
plt.show()

# Add Higher_Salary column before saving
df['Higher_Salary'] = df['Salary'].apply(lambda sal : 1 if sal > salary_mean else 0)
#print("\n# of records between below vs above average salaries:")
#print( df.groupby('Higher_Salary')['Higher_Salary'].count() )
proj_lib.master_df['Higher_Salary'] = proj_lib.master_df['AverageSalary'].apply(highsal)


####################################################### 
# For Title , select the most relevant job titles and 
# create the categorial columns

# Saving the end results into file
proj_lib.master_df.to_csv(proj_lib.master_df.name, index=False, encoding='utf-8')
print(f"\nFile have been saved as {proj_lib.master_df.name}")

####################################################### 
# observation on Data Analyst
# combining Data Scientist and Data Analyst into one posting found
# and top record came from hourly rated , not annual rate actually
Top10_barplot(df, 'Data Analyst')

                        
print(proj_lib.master_df[['Title','Salary']].iloc[997])
print('this highest record originated from hourly rate')

####################################################### 
# Which industries pays the highest for data scientst ?
# Looking at the data scientist, insurance industry pays the most
# range from $140k to $275k on an annual basis, 
# However it is not easy to identify the industry exhaustively 
Top10_barplot( df, 'Data Scientist')
                        
print(proj_lib.master_df[['Title','Salary']].iloc[993])
print('this highest record was identified from insurance industry')

####################################################### 
# Is there any growth in the Data Engineer jobs ?
# There is no career growth in this field as it seems the rate very close
# regardless of contract or intern type or permanent job
Top10_barplot(df, 'Data Engineer')
print('Average salary about the same for Intern/Contract/Perm')


####################################################### 
#EDA for location and summary :
#
#There is no numberic data, jump to model preparation to find features columns
#
#Possible work:
#  Split location into City, State
#
#Exceptions :
#  Not all records has State. Some location does not have City.
#
#Outliers:
#  Can come back to this part, find all the outliers and remove them.

