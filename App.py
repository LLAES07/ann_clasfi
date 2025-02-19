import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from deploy import objects_loadings
from tensorflow.keras.models import  load_model



# Cargado los pkls y el modelo 

model = load_model('model.h5')
label_encoder_age = objects_loadings('label_encoder.pkl')
onehot_geo = objects_loadings('onehot_encoder.pkl')
scaler = objects_loadings('scaler_features.pkl')



st.title('Predicción abandono de clientes')



# Inputs 

geography = st.selectbox('Geography', onehot_geo.categories_[0])

gender = st.selectbox('Género', label_encoder_age.classes_)

age = st.slider('Edad', 18, 92)

balance = st.number_input('Balance')

credit_score = st.number_input('Puntaje Crediticio')

estimated_salary = st.number_input('Salario estimado')

tenure = st.slider('Antiguedad', 0 , 10)

n_products = st.slider('Cantidad de productos', 1, 4)

has_credit_card = st.selectbox('Posee tarjeta de credito ?', [0, 1])

active_user = st.selectbox('El cliente está activo ? ', [0,1])

datos = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_age.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [n_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [active_user],
    'EstimatedSalary': [estimated_salary]
})


geo_encoed = onehot_geo.transform([[geography]])

geo_df = pd.DataFrame(geo_encoed, columns= onehot_geo.get_feature_names_out(['Geography']))
df = pd.concat([datos, geo_df], axis=1)


df_scaled = scaler.transform(df)

pred = model.predict(df_scaled)

pred_proba = pred[0][0]


if pred_proba > 0.5:
    st.write('Lo mas probable que el cliente se vaya')
else:
    st.write('El cliente tiene baja probabilidad de irse')