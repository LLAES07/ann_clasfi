import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import  load_model


def objects_loadings(filepath):

    with open(filepath, 'rb') as file:
        object = pickle.load(file)
        return object
    


if __name__ =='__main__':

    label_encoder_age = objects_loadings('label_encoder.pkl')
    onehot_geo = objects_loadings('onehot_encoder.pkl')
    scaler = objects_loadings('scaler_features.pkl')
    model = load_model('model.h5')

    datos = {
        'CreditScore': 45,
        'Geography': 'Spain',
        'Gender': 'Female',
        'Age': 35,
        'Tenure': 2,
        'Balance': 45000,
        'NumOfProducts': 1,
        'HasCrCard': 0,
        'IsActiveMember': 1,
        'EstimatedSalary': 42000
    }

    geo_encoed = onehot_geo.transform([[datos['Geography']]])
    print(geo_encoed)

    geo_df = pd.DataFrame(geo_encoed, columns= onehot_geo.get_feature_names_out(['Geography']))
    print(geo_df)


    df = pd.DataFrame([datos])
    
    df['Gender'] = label_encoder_age.transform(df['Gender'])

    df = pd.concat([df, geo_df], axis=1).drop(columns='Geography')

    print(df)

    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)

    if pred > 0.5:
        print('Lo mas probable que el cliente se vaya')
    else:
        print('El cliente tiene baja probabilidad de irse')