import streamlit as st
import os
import pandas as pd
import joblib as jb
heading_style = '''
<div style="color:red;" align='center'>
<h1>Diabetes-dataset</h1>
</div>
'''
def return_df(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    kbn={
    'Pregnancies':[Pregnancies],
    'Glucose':[Glucose],
    'BloodPressure':[BloodPressure],
    'SkinThickness':[SkinThickness],
    'Insulin':[Insulin],
	'BMI':[BMI],
	'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
    'Age':[Age]
     }   
    final_df=pd.DataFrame(kbn)
    return final_df
def base_model():
    bmodel=jb.load(os.path.join('diabetes_model.pkl'))
    return bmodel
st.markdown(heading_style, unsafe_allow_html=True)
Pregnancies=st.number_input('How much money was spent on Pregnancies', min_value=0)
Glucose=st.number_input('How much money was spent on Glucose', min_value=0)
BloodPressure=st.number_input('How much money was spent on BloodPressure',min_value=0)
SkinThickness=st.number_input('How much money was spent on SkinThickness',min_value=0)
Insulin=st.number_input('How much money was spent on Insulin', min_value=0)
BMI=st.number_input('How much money was spent on BMI', min_value=0)
DiabetesPedigreeFunction=st.number_input('How much money was spent on DiabetesPedigreeFunction',min_value=0)
Age=st.number_input('How much money was spent on Age',min_value=0)
df=return_df(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	st.write(predictions)
