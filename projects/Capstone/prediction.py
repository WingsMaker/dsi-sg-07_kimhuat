import flask
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os, sys, string, time
#from sklearn.tree import DecisionTreeClassifier

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
class ReusableForm(Form):
    depart = TextField('depart', validators=[validators.required(), validators.Length(min=3, max=3)])
    arrival = TextField('arrival', validators=[validators.required(), validators.Length(min=3, max=3)])
    schedule = TextField('schedule:', validators=[validators.required(), validators.Length(min=10, max=35)])
 

@app.route("/", methods=['GET', 'POST'])
def FlightArrivals():
    form = ReusableForm(request.form)
 
    if request.method == 'POST':
        depart = request.form['depart']
        arrival = request.form['arrival']
        schedule = request.form['schedule']
        prob = predict_delay(schedule, depart, arrival)
        results = 'predicted probability is ' + str( prob )
        print(results)
        flash(results)
        msg = 'The predicted probability for flight arrival on time is ' + str(prob) 
        result = "<script>alert('" + msg + "');</script>"
        return result + render_template('index.html', form=form)
    return render_template('index.html', form=form)


def predict_delay(xDepart = '1/10/2018 21:45:00', xFrom = 'JFK', xTo = 'ATL'):
    # example : predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')

    f = open('delaypredictor.pkl', 'rb')
    model = pickle.load(f)
    f.close() 

    try:
        #departure_date_time_parsed = datetime.strptime(xDepart, '%d/%m/%Y %H:%M:%S')
        departure_date_time_parsed = datetime.strptime(xDepart, '%Y-%m-%dT%H:%M')
    except ValueError as e:
        print('Error parsing date/time - {}'.format(e))
        return -1
    
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
        prob = model.predict_proba(pd.DataFrame(input))
        prob = prob[0][0]
    except:
        print('unable to perform preduction')
        prob = -1
    return prob

if __name__ == "__main__":
    app.run()
