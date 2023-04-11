# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:32:13 2023

@author: HP
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

PRmodel = pickle.load(open("PRmodel.pkl","rb"))

st.title('MODEL DEPLOYMENT : PROPHET-TIMESERIES FORECASTING')

st.sidebar.header('User input parameters')

def User_input_features():
    Date = st.sidebar.selectbox('PJMW_MW',('1','0'))
    Time = st.sidebar.selectbox('PJMW_MW',('1','0'))
    
    data = {'Date': Date,
            'Time': Time}
    
    features = pd.DataFrame(data,index=[10])
    
    return features

df = User_input_features()
st.subheader('User input parameters')
st.write(df)

MW_df=pd.read_excel('PJMW_MW_Hourly.xlsx')

MW_df.sort_values('Datetime')
MW_df = MW_df.sort_values('Datetime')
from prophet import Prophet

MW_df.columns = ['ds','y']

PRmodel = Prophet()
PRmodel.fit(MW_df)

### Create future dates of 30 days
future_dates=PRmodel.make_future_dataframe(periods=90)

prediction=PRmodel.predict(future_dates)
prediction

st.subheader('Predicted Result')



st.subheader('Prediction probability')
st.write(prediction)
