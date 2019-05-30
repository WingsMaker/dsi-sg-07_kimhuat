#!/usr/bin/env python
# coding: utf-8
# ____ ___   ____        _
#| __ )_ _| | __ )  ___ | |_
#|  _ \| |  |  _ \ / _ \| __|
#| |_) | |  | |_) | (_) | |_
#|____/___| |____/ \___/ \__|
#
#
from IPython.display import Image  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pydotplus
import pickle

import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals.six import StringIO  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from nltk.chat.eliza import eliza_chatbot
from chatbot import Chat,reflections,multiFunctionCall
import nltk, wikipedia

from datetime import datetime
import os, sys, string, random, time
import json, webbrowser
from sqlalchemy import create_engine

# telegram api
import asyncio
import telepot, telepot.aio
from telepot.loop import MessageLoop
from telepot.aio.loop import MessageLoop as Message_Loop
from telepot.aio.delegate import pave_event_space, per_chat_id, create_open

from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore")

# global variables
global TelegramBot, bi_bot_info, chat_id, chat_msg, runbot # chatbot related
global sent_tokens                             # NLP related
global dbengine, dw_index , bwdata, select_mode 

global datawarehouse
"""
ID      : index key
Name    : name of the datasource for serving "/source" command
URL     : hyperlink of the datasource location
Info    : brief information for serving "/source" command
Explain : some extra information for serving "/explain" command
"""

global tables 
"""
ID      : index key
Name    : name of datasource refered to
SQL     : actual SQL query for the connected database 
Table   : name of the table for serving "/tables" command
"""

global queries  
"""
ID      : index key
Name    : name of datasource refered to
SQL     : actual SQL query for the connected database or DataFrame
Info    : brief information for serving "/queries" command
Plot    : <x-axis variable>;<y-axis variable>;<bar,barh,etc>
"""

global reports
"""
ID      : index key
Name    : name of datasource refered to
Info    : brief information for serving "/queries" command
Report  : Name of report as well as the function name
"""

class MessageCounter(telepot.aio.helper.ChatHandler):
    def __init__(self, *args, **kwargs):
        super(MessageCounter, self).__init__(*args, **kwargs)
        self._count = 0
        return

    async def on_chat_message(self, msg):
        return
    
## libary functions
def handle(telegram_msg):    
    global TelegramBot, sent_tokens, select_mode, chat_id, chat_msg, dw_index, runbot

    content_type, chat_type, chat_id = telepot.glance(telegram_msg)    
    if content_type=='contact':
        name = telegram_msg['message']['contact']['first_name']
        phone = telegram_msg['message']['contact']['phone_number']
        user_id = telegram_msg['message']['contact']['user_id']
        msg =  'contact : ' + name + "," + str(user_id) + "," + phone
        print(msg)
        msgout(msg)
        fn = name + '.json'
        with open(fn, 'w') as outfile:  
            json.dump(telegram_msg, outfile)
        return

    chat_msg = telegram_msg['chat']
    user_response = telegram_msg['text']    
    user_response=user_response.lower()
    user_words = user_response.split(' ')
    list_func = ''
    opt = 0
    opt, list_func = scan_keywords( user_words )    
    if ('bye' in user_words) or (user_response == '/end') or (user_response == '/stop'):
        msg = "bi_bot: Bye! take care.."
        msgout(msg)
        clear_output()
        select_mode = 0
        if user_response == '/stop':
            runbot = 0
        exit
    elif ('thank' in user_words) or ('thx' in user_words):
        msg = "bi_bot: You are welcome.."
        msgout(msg)
        return
    elif (list_func != '') and (opt > 0):
        select_mode = opt
        msg = eval(list_func)
        msgout(msg)
        return
    elif (user_response[0]=='/'):        
        code_args = user_response[1:]
        code_options = code_args.split(' ')
        code_argcnt = len(code_options)
        if code_argcnt==0:
            return
        cmd_list = ['start', 'db', 'tbl', 'qry', 'rpt', 'pre']
        code_list = ['start_msg', 'ListDataSource', 'ListTables', 'ListQueries', 'ListReports', 'ListPredictions']
        select_mode = cmd_list.index(code_options[0])
        if (code_argcnt >= 2) and (select_mode > 0):
            handle_selection(code_options[1])
            return
        if (code_argcnt == 1) and (select_mode >= 0):
            code_to_run = code_list[select_mode] + '()'
            if code_to_run != '':
                try:
                    msg = eval(code_to_run)
                except:
                    msg = 'unable to process the above command'
                msgout(msg)
        return
    elif (user_response[:7]=='predict'):
        pred_func = get_pred_func()        
        code_to_run = pred_func + '(' + user_response[7:] + ')'
        msgout(code_to_run)
        try:
            msg = 'Predicted result : ' + eval(code_to_run)
        except:
            msg = 'unable to process the above command'
        msgout(msg)
    elif (user_response.isdigit()):
        handle_selection(user_response)
        return
    elif ('who' in user_words) or ('what' in user_words):
        words = [w for w in user_words if w not in ['what', 'who', 'where', 'is', 'are', 'the']]
        question = ' '.join(words)
        answer = whoIs(question)
        msg = "bi_bot: " + answer
        msgout(msg)            
        return
    else:
        if(greeting(user_response)!=None):
            msg = "bi_bot: " + greeting(user_response)
            msgout(msg)
        else:
            msg = get_response(user_response)                
            print(msg)
            msgout(msg)
            sent_tokens.remove(user_response)
        return                
    return

def scan_keywords(user_words_list):
    idx = 0
    func_id = 0
    func_name = ''
    list_func_assigned = ['DataSource', 'Tables', 'Queries', 'Reports', 'Predictions']
    for keyword in ['database', 'tables', 'queries', 'reports', 'predictions']:
        if keyword in user_words_list:
            func_name = 'List' + list_func_assigned[idx] + '()'
            func_id = idx + 1
            return func_id, func_name
        idx += 1
    return func_id, func_name

def get_pred_func():
    global datawarehouse, dw_index
    pred_func = ''
    dsname = datawarehouse['Name'][dw_index]
    cnt = len(reports['Name'])
    for n in range(cnt):
        id = n + 1
        rptname = reports['Name'][id]
        pred_opt = reports['Predict'][id]
        if pred_opt != pred_opt:
            pred_opt = ''
        if rptname == dsname:            
            if pred_opt != '':
                pred_func = pred_opt
                return pred_func
    return pred_func

def LemTokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def get_response(user_response):
    global sent_tokens
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        if(user_response[:6] == 'who is' or user_response[:7] == 'what is' or user_response[:7] == 'tell me'):
            question =user_response[5:] 
            robo_response = whoIs(question)
            return robo_response            
        #robo_response = robo_response + "I am sorry! I don't understand you"
        robo_response = robo_response + eliza_chatbot.respond(user_response)
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

def handle_selection(selection):
    global datawarehouse, dw_index, select_mode, tables, queries, reports
    
    if selection == '':
        return
    opt = 0   ;   valstr = selection.rstrip()
    if valstr.isdigit():
        opt = int(valstr)
    if opt == 0:
        return
    if select_mode == 1:
        if (opt > 0) and (opt <= len(datawarehouse)):
            msg = f'you have selected datasource from option {opt}'
            msgout(msg)
            dw_index = opt
            if ConnectDataWarehouse():
                msg = datawarehouse['Explain'][dw_index]
            else:
                msg = 'Remote database is not available, please try later'
            msgout(msg)
    elif select_mode == 2:
        if (opt > 0) and (opt <= len(tables['Name'])):
            msg = f'you have selected option {opt}'
            msgout(msg)
            msg = run_query(opt)
            msgout(msg)
    elif select_mode == 3:
        if (opt > 0) and (opt <= len(queries['Name'])):
            msg = f'you have selected option {opt}'
            msgout(msg)
            msg = run_query(opt)
            msgout(msg)
            fn = 'bi_chart.png'
            if plotgraph(opt, fn):
                SendPic(fn)
    elif select_mode == 4:
        if (opt > 0) and (opt <= len(reports['Name'])):
            msg = f'you have selected option {opt}'
            msgout(msg)
            pred_ana = reports['Report'][opt]       
            if pred_ana != pred_ana:
                pred_ana = ''
                print('the analytics module is empty!')
                return
            try:
                pred_func= pred_ana + '()'
                eval(pred_func)
                return
            except:
                msg = 'Unable to run the analysis , code = ' + pred_func
            msgout(msg)
    elif select_mode == 5:
        if (opt > 0) and (opt <= len(reports['Name'])):
            msg = f'you have selected option {opt}'
            msgout(msg)
            pred_func = reports['Predict'][opt]       
            if pred_func != pred_func:
                pred_func = ''
                print('the analytics module is empty!')
                return
            try:
                usage = reports['Usage'][opt]
                msg = 'Example on how to predict\npredict '+usage+'\n'
                msgout(msg)
                return
            except:
                msg = 'Unable to run the prediction , code = ' + pred_func
            msgout(msg)
    elif (select_mode >5) or (select_mode <1):
        msg = "Sorry, the selection doesn't make sense in the case."
        msgout(msg)
        select_mode = 0
    return
    
def ListDataSource():
    global datawarehouse, dw_index 
    n = len(datawarehouse['Name'])
    msg = f'you have {n} data source(s) to choose:\n'
    opt = ''
    for i in range(n):
        j = i + 1
        msg = msg + '#' + str(j) + ' : ' + datawarehouse['Name'][j] + '\n'
        msg = msg + 'Description : ' + datawarehouse['Info'][j] + '\n'
    msg = msg + 'reply 1 for first one, 2 for the second, and so on\n\n'
    msg = msg + 'current data source is : [' + str(dw_index) + ']'
    return msg

def ListQueries():
    global datawarehouse, dw_index, queries
    dsname = datawarehouse['Name'][dw_index]
    cnt = len(queries['Name'])
    result = 'list of queries for this data source:\n'
    for n in range(cnt):
        id = n + 1
        qname = queries['Name'][id]
        plt_opt = queries['Plot'][id]
        if plt_opt == plt_opt:
            opt = 1
        else:
            opt = 0
        plt_opt = 'Has a graph : ' + ('NO' if opt==0 else 'YES')
        if qname == dsname:
            qinfo = queries['Info'][id]
            if qinfo != qinfo:
                continue  # nan, no value found, take it not-valid query
            else:
                print(id, ':',qinfo)
                result = result + '#' + str(id) + ' ' + qinfo + '\n' + plt_opt + '\n'
                
    result = result + '\nTo perform the query #n, reply "n"\nwhere "n" is the query id'
    return result

def ListTables():
    global datawarehouse, dw_index, tables
    dsname = datawarehouse['Name'][dw_index]
    cnt = len(tables['Name'])
    result = 'list of tables for this data source:\n'
    for n in range(cnt):
        id = n + 1
        qname = tables['Name'][id]
        if qname == dsname:
            tblname = tables['Table'][id]
            result = result + '#' + str(id) + ' ' + tblname  + '\n'
    result = result + '\nTo view the table #n, reply "n"\nwhere "n" is the table id'
    return result

def ListReports():
    global datawarehouse, dw_index, reports
    dsname = datawarehouse['Name'][dw_index]
    cnt = len(reports['Name'])
    result = 'list of predictice analytics reports for this data source:\n'
    for n in range(cnt):
        id = n + 1
        rptname = reports['Name'][id]
        pred_ana = reports['Report'][id]
        if pred_ana != pred_ana:
            continue  # nan, no value found, take it not-valid query
        if rptname == dsname:
            rinfo = reports['Info'][id]
            if rinfo != rinfo:
                continue  # nan, no value found, take it not-valid query
            print(id, ':',rinfo)
            result = result + '#' + str(id) + ' ' + rinfo + '\n'
                
    result = result + '\nTo perform the analytics #n, reply "n"\nwhere "n" is the query id'
    return result

def ListPredictions():
    global datawarehouse, dw_index, reports
    
    dsname = datawarehouse['Name'][dw_index]    
    cnt = len(reports['Name'])
    result = 'list of predictice analytics reports for this data source:\n'
    for n in range(cnt):
        id = n + 1
        rptname = reports['Name'][id]
        pred_opt = reports['Predict'][id]
        if pred_opt != pred_opt:
            pred_opt = ''
        if rptname == dsname:            
            if pred_opt == '':
                continue  
            print(id, ':',pred_opt)
            result = result + '#' + str(id) + ' ' + pred_opt + '\n'
                
    result = result + '\nTo perform the analytics #n, reply "n"\nwhere "n" is the id number'
   
    return result


def run_query(id):
    global dbengine, dw_index, tables, queries, bwdata
    maxrow = 10
    if dw_index == 1:    
        if select_mode == 2:
            query = tables['SQL'][id]
            if query != query:  # nan
                query = 'Select * From ' + tables['Table'][id] + ' LIMIT 10'
        elif select_mode == 3:
            query = queries['SQL'][id]
        else:
            query = ''
        if query != '':
            bwdata = pd.read_sql(query, con = dbengine)
        if select_mode == 2:
            df_text = tables['Table'][id] + '\n\n'
        else:
            df_text = queries['Info'][id] + '\n\n'
        df_text = df_text +  str(list(bwdata.columns)) + '\n'
        cnt = 0
        for i in bwdata.index:
            row = list(bwdata.loc[i])
            df_text = df_text + str(row) + '\n'
            cnt += 1
            if cnt >= maxrow:
                return df_text
        return df_text
    if (dw_index == 2) or (dw_index == 3):
        if select_mode == 2:
            cnt = bwdata.shape[0]
            df = eval("bwdata.head(10)")
            df_text = str(list(bwdata.columns)) + '\n'
            for i in df.index:
                row = list(df.loc[i])
                df_text = df_text + str(row) + '\n'
            df_text = df_text + f'total = {cnt} records'
            return df_text
        if select_mode == 3:
            query = queries['SQL'][id]
            if query!=query:
                query = ''
            if query=='':
                df = eval("bwdata")
            else:
                try:
                    df = eval(query)
                except:
                    print('there is an error on query, check:\n')
                    print(query)
                    return '' 
            return str(df.head(10))
    return ''

def ConnectDataWarehouse():
    global bi_bot_info, datawarehouse, dbengine, dw_index, bwdata, tables, queries, reports

    config_filename = bi_bot_info['config']
    datawarehouse =  pd.read_excel(open(config_filename, 'rb'), sheet_name='source', index_col='ID' ).to_dict() 
    tables =  pd.read_excel(open(config_filename, 'rb'), sheet_name='tables', index_col='ID' ).to_dict() 
    queries =  pd.read_excel(open(config_filename, 'rb'), sheet_name='queries', index_col='ID' ).to_dict() 
    reports =  pd.read_excel(open(config_filename, 'rb'), sheet_name='reports', index_col='ID' ).to_dict() 
    dsname = datawarehouse['Name'][dw_index]
    datasource = datawarehouse['URL'][dw_index]
    dbengine = None
    print('Current datasource is : ', dsname)
    if datasource.find('postgresql:') >=0 :
        if dbengine == None:
            print('connecting to remote database via SQL engine')        
            try:
                dbengine = create_engine(datasource)
                try:
                    bwdata = pd.read_sql('select * from products', con = dbengine)
                    print('connected to remote database')                    
                except:
                    print('unable to connect remote database')
                    return False
            except:
                dbengine = None
                print('unable to connect remote database')
                return False
    elif datasource.find('http:') >=0 :
        print('mounting remote datafile as dataframe')
        dbengine.dispose()
        dbengine == None
        bwdata = pd.DataFrame
        bwdata.name = dsname
        bwdata = pd.read_csv(datasource)                
    elif datasource.find('.csv') >=0 :
        print('using local datafile as dataframe')
        dbengine == None
        bwdata = pd.DataFrame
        bwdata.name = dsname
        bwdata = pd.read_csv(datasource)         
    return True

def plotgraph(opt, fname = ''):
    global bwdata, dw_index
    
    chart_data = queries.get('Plot').get(opt)
    if chart_data != chart_data:
        print('The plotgraph function is not available.')
        return False
    query = queries.get('SQL').get(opt)
    if query != query:
        print('The plotgraph function is not available.')
        return False
    if dw_index==1:
        query = 'bwdata'
    try:
        df = eval(query)
    except:
        print('There is error processing the query : ', query)
        print('\nPlease check with the support team on query id')
        return False
    title = queries.get('Info').get(opt)
    print(chart_data)
    chart_info = chart_data.split(';')
    print(chart_info)
    xvar = chart_info[0]
    yvar = chart_info[1]
    chart_type = chart_info[2]
    if chart_type=='boxplot':
        ax = df.boxplot(column=xvar, by=yvar)
    else:
        chart_type = 'plot.' + chart_type
        fn = "df." + chart_type + "(rot = 0,x = xvar,y = yvar,title = title, legend=xvar)"
        try:
            ax=eval(fn)
        except:
            print('unable to perform the plot for ' + fn)
            return False
        showvalues=[ax.text(i.get_width(), i.get_y(),str(i.get_width()))for i in ax.patches]
        plt.title(title)
    if fname != '':
        plt.draw()
        plt.savefig(fname, dpi=100)
        print(f'graph saved as {fname}')
    plt.show()    
    return True

def SendPic(fname = ''):
    global TelegramBot
    if fname == '':
        return
    f = open(fname, 'rb')
    response = TelegramBot.sendPhoto(chat_id, f)
    return response

def start_msg():
    global chat_msg
    
    user_name = chat_msg['first_name']   
    msg = 'Hi ' + user_name + ' !\n'
    msg  = msg + 'My name is Theia, I am a Chatbot for BI.\n'
    msg  = msg + 'You can use "/" commands to obtain information.\nIf you want to exit, type "bye"\n\n'
    msg  = msg + 'Online predictor is also available at https://sites.google.com/view/bi-bot/online-predictor'
    return msg

def AnovaTest():
    global dbengine, dw_index, bwdata, reports

    if dw_index != 1:
        return
    id = 1
    report = reports['Intro'][id]
    fname = 'bi_chart.png'
    qry = """
    select c."Region" , d."OrderID" , d."Quantity" , d."UnitPrice" , d."Discount"
    from Customers as c inner join orders as o on c."CustomerID" = o."CustomerID"
    inner join order_details as d on d."OrderID" =  o."OrderID"
    where c."Region" is not NULL
    """

    # loading the order details with customer region categories
    df = pd.read_sql(qry, con = dbengine)

    # Calculating the revenue per sub-order
    df['price_per_order'] = df.Quantity * df.UnitPrice * (1 - df.Discount)

    # Dropping the columns for quantity, unit price and discount now that we have the total revenue
    df.drop(['Quantity', 'UnitPrice', 'Discount'], axis=1, inplace=True)

    # Grouping the data by order and summing the revenue for each order
    bwdata = df.groupby(['Region', 'OrderID'])['price_per_order'].sum().reset_index()

    # Plotting the distributions for the data
    plt.figure(figsize=(8,5))
    for region in set(df.Region):
        region_group = df.loc[df['Region'] == region]
        sns.distplot(region_group['price_per_order'], hist_kws=dict(alpha=0.2), label=region)
    plt.legend()
    plt.xlabel('Price per order')    
    plt.savefig(fname, dpi=100)
    print(f'graph saved as {fname}')
    plt.draw()
    plt.show()    
    
    # Fitting a model of price_per_order on Region categories, 
    # and using statsmodels to compute an ANOVA table
    try:
        f = open('pricing_model.pkl', 'rb')
        pricing_model = pickle.load(f)
        f.close()    
    except:
        print('unable to load from pricing model from pickle')
        #pricing_model = ols('price_per_order ~ C(Region)', bwdata).fit()
        return 'Missing file pricing_model.pkl'

    pricing_aov_table = sm.stats.anova_lm(pricing_model, typ=2)
    
    print('\nAnova table:')
    print(pricing_aov_table)
    
    report = report + 'Anova Table:\n'
    list_cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']
    report = report + '\t'.join(list_cols) + '\n'
    for i in pricing_aov_table.index:
        report = report + i + ':'
        row = list(pricing_aov_table.loc[i])
        report = report + str(row) + '\n'
    
    result = pricing_model.summary2()
    print('\nStats Model Summary:')
    print(result)    
    report = report + str(result) + '\n\n'
    
    print('\n\nConclusion (Hypothesis Test Result):\n')
    report = report + '\n\nHypothesis Test Result:\n'
    if pricing_aov_table.iloc[0,3] <0.05:
        print('We can reject null hypothesis')
        report = report + '\n\nWe can reject null hypothesis\n'        
        report = report + '\n\nThe average amount spent per order between regions is the different\n'
    else:
        print('We cannot reject null hypothesis')
        report = report + '\n\nWe cannot reject null hypothesis\n'
        report = report + '\n\nThe average amount spent per order between regions is the same\n'

    msgout(report)
    SendPic(fname)
    return 

def GPA_Test():    
    global dw_index, bwdata, reports
    if dw_index != 2:
        return
    id = 3
    report = reports['Intro'][id]
    
    Xr = bwdata[['admit','gre','rank']]
    yr = bwdata.gpa.values
    
    reg_scores = cross_val_score(LinearRegression(), Xr, yr, cv=4)    
    print('Validation Score for linear regression model:\n', reg_scores, np.mean(reg_scores))
    
    linreg = LinearRegression().fit(Xr, yr)
    linreg_model_score = linreg.score(Xr,yr)
    print('LinearRegression test score :', linreg_model_score)
    
    dtree_reg = DecisionTreeRegressor(max_depth=4)
    adaboost_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=8)
    dtree_reg.fit(Xr, yr)
    adaboost_reg.fit(Xr, yr)
    
    dtr_scores = cross_val_score(dtree_reg, Xr, yr, cv=4)
    dtr_model_score = dtree_reg.score(Xr, yr)
    
    adb_scores = cross_val_score(adaboost_reg, Xr, yr, cv=4)
    adb_model_score = adaboost_reg.score(Xr, yr)

    report = report + '\n'
    report = report + 'Linear Regression      Model Score :' + str(linreg_model_score) + '\n'
    report = report + 'Decsion Tree Regressor Model Score :' + str(dtr_model_score) + '\n'
    report = report + 'AdaBoost with Decision Tree  Score :' + str(adb_model_score) + '\n'
    report = report + '\n\n' + str(bwdata.head(5)) + '\n'
    report = report + '\n\nConclusion: ML model adaboost_reg created to predict gpa score.\n'

    print(report)
    msgout(report)
    fname = 'bi_chart.png'
    dot_data = StringIO()
    export_graphviz(dtree_reg, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names = ['admit','gre','rank'])  

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_png(fname)
    display(Image(graph.create_png()))
    SendPic(fname)
    return

class DTreeClass(object):
    def __init__(self):
        self.name = ''
        self.cvscore = 0
        self.modelscore = 0
        self.fp = 0   # false_positive_rate
        self.tp = 0   # true_positive_rate
        self.thres    # thresholds
        self.false_positive_rate = 0
        self.false_positive_rate = 0
        self.roc_auc = pricing_model
        self.confusion_matrix = None
        self.dtree = None
        pass

    def __init__(self, tgt, features):
        global dw_index, dtree_cls, bwdata, reports
        self.name = 'dtree'
        self.dtree = DecisionTreeClassifier(max_depth=3)
        Xc = bwdata[features]
        yc = bwdata[tgt].values        
        x_train, x_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.25)
        self.dtree.fit(x_train, y_train)
        self.cvscore = cross_val_score(self.dtree, x_train, y_train, cv=4)
        self.modelscore = self.dtree.score(x_train, y_train)
        y_pred = self.dtree.predict(x_test)
        self.fp, self.tp, self.thres = roc_curve(y_test, y_pred)
        self.roc_auc = auc(self.fp, self.tp)  
        self.confusion_matrix = confusion_matrix(y_test, y_pred)

    def predict(self, X):
        return self.dtree.predict(X)
        
    def plot_and_send(self, fname):
        plt.plot(self.fp, self.tp)
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC Score ' + str(self.roc_auc))
        plt.savefig(fname, dpi=100)
        plt.draw()
        plt.show()
        SendPic(fname)

def Product_Continuity():
    global dw_index, dbengine, bwdata, reports
    if dw_index != 1:
        return
    id = 2
    report = reports['Intro'][id]
    qry = """
    Select "Discontinued","UnitPrice","UnitsInStock","UnitsOnOrder","ReorderLevel"
    from Products
    """
    bwdata = pd.read_sql(qry, con = dbengine)
    try:
        dtree_product = DTreeClass('Discontinued', ["UnitPrice","UnitsInStock","UnitsOnOrder","ReorderLevel"])
    except:
        print('Error to initiate DTreeClass object to this model\n')
        
    report = report + '\n'
    report = report + 'Decsion Tree Regressor Model Score :' + str(dtree_product.modelscore) + '\n'
    report = report + 'ROC AUC Score :' + str(dtree_product.roc_auc) + '\n'
    
    roc_auc = dtree_product.roc_auc
    cf_matr = dtree_product.confusion_matrix
    
    report = report + str(cf_matr)
    report = report + '\n\n' + str(bwdata.head(5)) + '\n'
    report = report + '\n\nConclusion: ML model dtree_product created to predict admit value.\n'
    print(report)
    msgout(report)
    
    fname = 'bi_chart.png'
    try:
        dtree_product.plot_and_send(fname)    
    except:
        print('unable to send graph')
    return
        
def Admit_Test():
    global dw_index, bwdata, reports
    if dw_index != 2:
        return
    id = 4
    report = reports['Intro'][id]
    dtree_cls = DTreeClass('admit', ['gpa','gre','rank'])
    
    report = report + '\n'
    report = report + 'Decsion Tree Regressor Model Score :' + str(dtree_cls.modelscore) + '\n'
    report = report + 'ROC AUC Score :' + str(dtree_cls.roc_auc) + '\n'
    
    roc_auc = dtree_cls.roc_auc
    cf_matr = dtree_cls.confusion_matrix
    
    report = report + str(cf_matr)
    report = report + '\n\n' + str(bwdata.head(5)) + '\n'
    report = report + '\n\nConclusion: ML model dtree_cls created to predict admit value.\n'
    print(report)
    msgout(report)
    
    fname = 'bi_chart.png'
    dtree_cls.plot_and_send(fname)    
    return 

def FlightArr_Test():
    global TelegramBot, chat_id
    msgout('Flight arrival Analytics Report')
    try:
        status = TelegramBot.sendDocument(chat_id=chat_id, document=open('FlightArrivals.html', 'rb'))
        print(status)        
    except:
        return

def predit_admit(x_gre=520, x_gpa=3, x_rank=2):
    # example : predict x_gre=520, x_gpa=3, x_rank=2
    global bwdata
    
    dtree = DTreeClass('admit', ['gpa','gre','rank'])
    df = pd.DataFrame({'admit':[x_gre], 'gre':[x_gpa], 'rank':[x_rank]})
    list_admit = dtree.predict(df)
    admit_predicted = list_admit[0]        
    return str(admit_predicted)

def predit_discont(x_unitprice=20, x_stock=30, x_unitsonorder=0, x_reclevel=0):
    # example : predict x_unitprice=20, x_stock=10, x_unitsonorder=0, x_reclevel=0
    global bwdata,dbengine
    
    qry = """
    Select "Discontinued","UnitPrice","UnitsInStock","UnitsOnOrder","ReorderLevel"
    from Products
    """
    bwdata = pd.read_sql(qry, con = dbengine)    
    
    dtree = DTreeClass('Discontinued', ["UnitPrice","UnitsInStock","UnitsOnOrder","ReorderLevel"])
    df = pd.DataFrame({'UnitPrice':[x_unitprice], 'UnitsInStock':[x_stock], 'UnitsOnOrder':[x_unitsonorder], 'ReorderLevel':[x_reclevel]})
    list_discont = dtree.predict(df)
    print(df)
    discont_pred = list_discont[0]
    print(discont_pred)
    return str(discont_pred)

def predict_delay(xDepart = '1/10/2018 21:45:00', xFrom = 'JFK', xTo = 'ATL'):
    # example : predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')

    f = open('delaypredictor.pkl', 'rb')
    model = pickle.load(f)
    f.close()

    try:
        departure_date_time_parsed = datetime.strptime(xDepart, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)
    
    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour
    
    xFrom = xFrom.upper()
    xTo = xTo.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if xFrom == 'ATL' else 0,
              'ORIGIN_DTW': 1 if xFrom == 'DTW' else 0,
              'ORIGIN_JFK': 1 if xFrom == 'JFK' else 0,
              'ORIGIN_MSP': 1 if xFrom == 'MSP' else 0,
              'ORIGIN_SEA': 1 if xFrom == 'SEA' else 0,
              'DEST_ATL': 1 if xTo == 'ATL' else 0,
              'DEST_DTW': 1 if xTo == 'DTW' else 0,
              'DEST_JFK': 1 if xTo == 'JFK' else 0,
              'DEST_MSP': 1 if xTo == 'MSP' else 0,
              'DEST_SEA': 1 if xTo == 'SEA' else 0 }]
    try:
        result = model.predict_proba(pd.DataFrame(input))[0][0]
    except:
        print('unable to perform preduction')
    return str(result)

def msgout(msg, html_mode = 0):
    global TelegramBot, chat_id
    if msg == '':
        return
    try:
        if html_mode == 0:
            TelegramBot.sendMessage(chat_id, msg)
        else:
            TelegramBot.sendMessage(chat_id, parse_mode='HTML')
    except:
        return

def whoIs(query,sessionID="general"):
    try:
        return wikipedia.summary(query)
    except:
        for newquery in wikipedia.search(query):
            try:
                return wikipedia.summary(newquery)
            except:
                pass
    return eliza_chatbot.respond(query)

def bot_getconfig(configfile):
    with open(configfile) as json_file:  
        bi_bot_info = json.load(json_file)
    return bi_bot_info

def clear_chat(bot_token):
    bot = telepot.aio.DelegatorBot(bot_token, [
        pave_event_space()(
            per_chat_id(), create_open, None, timeout=5),
    ])
    endchat = Message_Loop(bot)
    print('chat history already cleared')

def input_raw():
    f = open('chatbot.txt','r',errors = 'ignore')
    return f.read().lower()

# starting up telegram bot with connection
bi_bot_info = bot_getconfig('bi_bot.json')
bot_token = bi_bot_info['token']
config_filename = bi_bot_info['config']

TelegramBot = telepot.Bot(bot_token)
status = TelegramBot.getMe()    
bot_id = status.get('id')   
bot_url = 'https://telegram.me/' + bi_bot_info['botuser']

dw_index = 1
if ConnectDataWarehouse():
    msg = datawarehouse['Explain'][dw_index]
else:
    msg = 'Remote database is not available, please try later'
msgout(msg)
select_mode = 0
chat_id = 0
load_nlp = False
if load_nlp:
    nltk.download() # for downloading packages
raw = input_raw()
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
#word_tokens = nltk.word_tokenize(raw)# converts to list of words

clear_output()
clear_chat(bot_token)
runbot = 1
print ('bi_bot is listening ...')

MessageLoop(TelegramBot, handle).run_as_thread()

# Keep the program running.
webbrowser.open(bot_url, new = 2)
while (runbot == 1):
    try:
        time.sleep(3)
    except:
        runbot = 0

try:
    os.kill(os.getpid(), 9)
except:
    print('Goodbye!')    