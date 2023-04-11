# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:32:13 2023

@author: HP
"""


import pickle
import pandas as pd
import streamlit as st


#@app.route('/')
def welcome():
    return "Welcome All"



pickle_in = open("PRmodel.pkl","rb")
PRmodel=pickle.load(pickle_in)

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(ds, yhat):
    
  prediction=PRmodel.predict([[ds,yhat]])
  print(prediction)
  return prediction

def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    Datetime = st.text_input("Datetime","Type Here")
    result=""


st.title('MODEL DEPLOYMENT : PROPHET-TIMESERIES FORECASTING')

st.sidebar.header('User input parameters')

def User_input_features():
    Date = st.sidebar.selectbox('Date',('1','0'))
    Time = st.sidebar.selectbox('Time',('1','0'))
    
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

### Create future dates of 90 days
future_dates=PRmodel.make_future_dataframe(periods=90)

prediction=PRmodel.predict(future_dates)
prediction

st.subheader('Predicted Result')


if st.button("Predict"):
   result=predict_note_authentication()
   st.success('The output is {}'.format(result))
if st.button("About"):
   st.text("Lets LEarn")
   st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    

