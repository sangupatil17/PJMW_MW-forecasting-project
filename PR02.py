# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 07:16:38 2023

@author: Sangamesh
"""


import pandas as pd
import streamlit as st 
from prophet import Prophet
import tqdm





#@app.route('/')
def welcome():
    return "Welcome All"

def main():
    st.title("PJMW_MW hourly-Electricity consumption")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit PJMW_MW_Electricity consumption App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

st.title("PJMW_MW-ELECTRICITY CONSUMPTION")
df = pd.read_excel('PJMW_MW_Hourly.xlsx', parse_dates=True)


def user_input_features():
    date_input = st.date_input("Enter a datetime")
    data = {'Datetime':date_input}
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)



## prophet model fitting
def create_fit_prophet_model(df):
    # Rename columns to ds and y
    df = df.rename(columns={'Datetime': 'ds', 'PJMW_MW': 'y'})
    # Create and fit Prophet model 
    PRmodel = Prophet()
    PRmodel.fit(df)
    return PRmodel

# Define function to make predictions using the Prophet model
def make_predictions(PRmodel, df, periods):
    future = PRmodel.make_future_dataframe(periods=periods)
    forecast = PRmodel.predict(future)
    return forecast[-periods:][['ds', 'yhat']]


# Create sidebar for selecting number of periods to forecast
periods = st.sidebar.slider('Number of periods to forecast', 1, 365)

PRmodel = create_fit_prophet_model(df)

# Make predictions using Prophet model
forecast = make_predictions(PRmodel, df, periods)

st.subheader('Predicted Result')
st.write(forecast)
























