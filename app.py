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
Pregnancies=st.number_input('Number of Pregnancies', min_value=0)
Glucose=st.number_input('Enter your Glucose levels', min_value=0)
BloodPressure=st.number_input('Enter your BloodPressure levels',min_value=0)
SkinThickness=st.number_input('Enter your SkinThickness',min_value=0)
Insulin=st.number_input('Enter number of Insulin', min_value=0)
BMI=st.number_input('Enter BMI', min_value=0)
DiabetesPedigreeFunction=st.number_input('Diabetes Pedigree Function',min_value=0)
Age=st.number_input('Enter your Age',min_value=0)
df=return_df(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	st.write(predictions)
