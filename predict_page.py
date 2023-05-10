import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb')as file:
        data=pickle.load(file)
    return data
data=load_model()


regressor=data['model']
le_country=data['le_country']
le_education=data['le_education']

def show_predict_page():
    st.title("Software developer salary prediction")
    


    
    
    st.write("""### We need some infomation to predict the salary""")

countries=('Australia',
            'Brazil',
            'Canada',
            'France',
            'Germany',
            'India',
            'Israel',
            'Italy',
            'Netherlands',
            'Other',
            'Pakistan',
            'Poland',
            'Russian Federation',
            'Spain',
            'Sweden',
            'Turkey',
            'United Kingdom',
            'United States')
education=(
    'less than a bachelor’s',
    'Bachelor’s degree',
    'Master’s degree',
    'post grad'
)

country=st.selectbox('Country',countries)
education=st.selectbox('education level',education)

experience=st.slider('Years of experience',0,50,3)
hours_worked=st.slider('Hours worked',0,480,1)
ok=st.button("Calculate Salary")
if ok:
    X=np.array([[country,education,experience,hours_worked]])
    X[:,0]=le_country.transform(X[:,0])
    X[:,1]=le_education.transform(X[:,1])
    X=X.astype(float)

    salary=regressor.predict(X)
    st.subheader(f"The estimated salary is ${salary[0]:.2f}")