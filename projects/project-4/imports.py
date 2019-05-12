import pandas as pd
from time import sleep
import sys, os, time
import warnings
import importlib
import proj_lib
                        
warnings.filterwarnings('ignore') 

####################################################### 
#  Main Program
####################################################### 
# this section search all related pages & calculates how many pages it takes into max_page
# topics of interest:
# data scientist, data engineer, business intelligence, data analyst ...

if len(sys.argv)<2:
    print('please specify which sites to scrap from.')
    sys.exit(0)
    
# local variables:


# constantly maintain page_url variable to scrap more data
#page_url = 'https://www.indeed.com/jobs?q=data+architect&start={}'
page_url = sys.argv[1]
print( page_url )

# The datafile is ready for analysis since web scraping is done
# we cab actual skip the load part
max_page = 990
datafile = 'indeed_df.csv'

max_page, lastpage = proj_lib.FindLastPage(page_url)
print(f"Last page url :\n{lastpage}")

# set to True to load from website, False to skip
# since all files downloaded, we can set it to False
LoadFromSite = False   
if LoadFromSite:
    try:        
        from selenium import webdriver
        
        chromedriver="./chromedriver/chromedriver"
        os.environ["webdriver.chrome.driver"] = chromedriver
        driver = webdriver.Chrome(executable_path=chromedriver)  
    except:
        driver = webdriver.Chrome()
    finally:
        proj_lib.LoadWebPages(driver, page_url, 0, max_page)    
    driver.close()        
    
# process all the saved web pages into dataframe indeed_df
VerifyMode = False   
print("verifying the files...")
if VerifyMode:
    max = max_page + 1
    startnum = 0
    for n in range( startnum, max, 10 ):
        fn = proj_lib.soupfile(n)
        if n % 100 ==0:
            print(fn)
        if proj_lib.process_webpage(fn, True) == False:
            print(f"Error process file {fn}")    
            break
print("Finished verifying soup files")

# Is there any files having incomplete records ? 
# Nope
# ( purpose also to uncover possible bugs )
if os.path.isfile(datafile):
    print("data file exists, loading it now")
    indeed_df = pd.read_csv(datafile)
else:
    indeed_df = pd.DataFrame({
    'Title' : [], 
    'Company' : [], 
    'City' : [], 
    'Salary' : [], 
    'Summary' : []
    })

# now we have verified data to process
# process all the saved web pages into dataframe indeed_df
startnum = 0
max = max_page + 1
for n in range( startnum, max, 10 ):
    fn = proj_lib.soupfile(n)
    if n % 100 ==0:
        print(fn)
    proj_lib.process_webpage(indeed_df, fn, False)

print("Finished processing")

# Saving the end results into file
indeed_df.to_csv(datafile, index=False, encoding='utf-8')
print(f"File have been saved as {datafile}")

proj_lib.master_df = indeed_df
