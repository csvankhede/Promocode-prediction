from flask import Flask, request, jsonify
import traceback
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import json_normalize
from datetime import datetime
import numpy as np


app = Flask(__name__)

def preprocess(json_):
    """
    This function preproces the data

    parameter
    ---------
    json_: json input

    returns:
    np.ndarray
    """
    df_targets = json_normalize(json_)
    df_targets['is_purchased'] = np.where(df_targets['event_type']=='purchase',1,0)

    # remove same product in same session by selecting only on record 
    df_targets['is_purchased'] = df_targets.groupby(['user_session','product_id'])['is_purchased'].transform('max')

    # drop duplicate of 'cart' category
    df_targets = df_targets.loc[df_targets['event_type']=='cart'].drop_duplicates(['user_session','product_id','is_purchased'])

    # add week day of the record
    df_targets['event_weekday'] = df_targets['event_time'].apply(lambda x:str(datetime.strptime(str(x)[:10],'%Y-%m-%d').weekday()))

    # add level 1 and 2 category
    df_targets["category_code_level1"] = df_targets["category_code"].str.split(".",expand=True)[0].astype('category')
    df_targets["category_code_level2"] = df_targets["category_code"].str.split(".",expand=True)[1].astype('category')
    # select users who add item to cart or purchase
    users = df_targets.loc[df_targets['event_type'].isin(['cart','purchase'])].drop_duplicates(subset=['user_id'])
    users.dropna(how='any',inplace=True)
    users_activity = df_targets.loc[df_targets['user_id'].isin(users['user_id'])]
    activity_in_session = users_activity.groupby(['user_session'])['event_type'].count().reset_index()
    activity_in_session.rename(columns={'event_type':'activity_count'},inplace=True)    
    df_targets['hour'] = df_targets['event_time'].apply(lambda x:str(datetime.strptime(str(x)[:-4],'%Y-%m-%d %H:%M:%S').hour))
    df_targets = df_targets.merge(activity_in_session, on='user_session', how='left')
    df_targets['activity_count'] = df_targets['activity_count'].fillna(0)
    features = df_targets.loc[:,['price','brand','event_weekday','category_code_level1','category_code_level2', 'activity_count','hour']]

    features['category_code_level1'] = features['category_code_level1'].astype('string')
    features['category_code_level2'] = features['category_code_level2'].astype('string')
    features['brand'].fillna('NaN',inplace=True)
    features['category_code_level1'].fillna('NaN',inplace=True)
    features['category_code_level2'].fillna('NaN',inplace=True)
    features['brand'] = features['brand'].astype('string')
    features['hour'] = features['hour'].astype('int')
    features['event_weekday'] = features['event_weekday'].astype('int')

    brand_lbl = pickle.load(open('brand_enc.pkl','rb'))
    cat1 = pickle.load(open('cat1_enc.pkl','rb'))
    cat2 = pickle.load(open('cat2_enc.pkl','rb'))
    features['brand'] = brand_lbl.transform(features['brand'])
    features['category_code_level1'] = cat1.transform(features['category_code_level1'])
    features['category_code_level2'] = cat2.transform(features['category_code_level2'])

    scaler = pickle.load(open('scaler.pkl','rb'))
    features_sc = scaler.transform(features)

    return features_sc


@app.route('/', methods=['GET','POST'])
def predict():
    try:
        json_ = request.get_json(force=True)
        features = preprocess(json_)
        pred = model.predict(features).tolist()
        if pred:
            return jsonify({'prediction':'No need for promo'})
        else:
            return jsonify({'prediction':'Give promo'})
    except:
        return jsonify({'trace': traceback.format_exc()})
    
if __name__ == '__main__':
    """
    Instructions
    ------------
    1) Run this file
    2) In another terminal run `./ngrok http 5000`
    
    example data point: 
    {"event_time":"2019-10-01 02:42:12 UTC","event_type":"cart","product_id":1005100,"category_id":2053013555631882655,"category_code":"electronics.smartphone","brand":"samsung","price":154.42,"user_id":555463605,"user_session":"d5a6dd40-851e-44b1-b53b-5f756e3205c0"}
    
    3) Access this REST API as below
    `curl -X GET [ngrok url] -d [json data point]

    example:
    `curl -X GET http://aa861b7d.ngrok.io/ -d '{"event_time":"2019-10-01 02:42:12 UTC","event_type":"cart","product_id":1005100,"category_id":2053013555631882655,"category_code":"electronics.smartphone","brand":"nokia","price":89.72,"user_id":555463605,"user_session":"d5a6dd40-851e-44b1-b53b-5f756e3205c0"}`

    returns
    -------
    json response

    example: {'prediction':'No need for promo'}
    """
    print('Loading model...')
    model = pickle.load(open('model.pkl','rb'))
    app.run(debug=True)
    # 
    # {"event_time":"2019-10-01 02:42:12 UTC","event_type":"cart","product_id":1005100,"category_id":2053013555631882655,"category_code":"electronics.smartphone","brand":"nokia","price":89.72,"user_id":555463605,"user_session":"d5a6dd40-851e-44b1-b53b-5f756e3205c0"}

