from typing import Tuple
from model import *
from yelp import *
from helper import *
from firebase import *

import pandas as pd
import numpy as np
import random
import tensorflow as tf
import random
import json
import base64

ENDPOINT = "s3://cdk-hnb659fds-assets-855388087552-ap-east-1/model_test.h5"
MODEL_LOCAL_PATH = "/tmp/model_test.h5"
RESULT_LOCAL_PATH = "/tmp/model_eval.csv"


def create_training_dataset(df_user, timestamp_trained):
    """
    Create dataset for model re-training
    Args:
        df_user (df): dataframe of all user profiles
        timestamp_trained: timestamp of last re-training
    Returns:
        df_hist (df): dataframe of dataset for model re-training
    """
    print("create re-training dataset")
    df_hist = pd.DataFrame(columns=['uid', 'rid', 'rating'])
    uid_list = []
    rating_list = []
    rid_list = []
    for i in range(df_user.shape[0]):
        if 'review' in df_user['history'][i]:
            for timestamp in df_user['history'][i]['review']:
                if int(timestamp) > timestamp_trained and \
                    "rating" in df_user['history'][i]['review'][timestamp] and \
                        "rid" in df_user['history'][i]['review'][timestamp]:
                    uid_list.append(df_user['uid'][i])
                    rating_list.append(df_user['history'][i]['review'][timestamp]['rating'])
                    rid_list.append(df_user['history'][i]['review'][timestamp]['rid'])
    
    df_hist = pd.DataFrame(list(zip(uid_list, rid_list, rating_list)), columns=['uid', 'rid', 'rating'])
    print("df_hist size: ", df_hist.shape[0])

    # retrieve restaurant profiles
    drop_r_list = []
    df_restaurant = pd.DataFrame(columns=['rid', 'rating'])
    for i in range(df_hist.shape[0]):
        if df_hist['rid'][i] != "":
            if df_hist['rid'][i] not in df_restaurant.reset_index()['rid'].tolist():
                df_one_r = get_restaurant_profile(df_hist['rid'][i])
                if df_one_r is not None:
                    df_restaurant = pd.concat([df_restaurant, df_one_r])
                else:
                    drop_r_list.append(df_hist['rid'][i])
        else:
            drop_r_list.append(i)
    df_hist.drop(drop_r_list, axis=0, inplace=True)
    df_hist = df_hist.reset_index(drop=True)
    df_restaurant = df_restaurant.reset_index(drop=True)

    # data preprocessing
    df_restaurant['price'] = df_restaurant['price'].apply(lambda x: len(x) if not pd.isnull(x) else 0.0)
    # df_restaurant['price'] = df_restaurant['price'].fillna(0.0)
    df_restaurant['rating'] = df_restaurant['rating'].fillna(0.0)
    df_restaurant['price'] = (df_restaurant['price'] - 1.0) / (3.0)
    df_restaurant['rating'] = (df_restaurant['rating'] - 1.0) / (4.0)
    df_restaurant['rid'] = df_restaurant['rid'].str.strip()

    df_hist['rating'] = (df_hist['rating'] - 1.0) / (4.0)
    df_hist['price'] = df_hist['rid'].apply(lambda x: df_restaurant[df_restaurant['rid'] == x]['price'].iloc[0] if x in df_restaurant['rid'].tolist() else 0)
    df_hist['r_categories'] = df_hist['rid'].apply(lambda x: df_restaurant[df_restaurant['rid'] == x]['categories'].iloc[0] if x in df_restaurant['rid'].tolist() else 0)
    df_hist['overall_rating'] = df_hist['rid'].apply(lambda x: df_restaurant[df_restaurant['rid'] == x]['rating'].iloc[0] if x in df_restaurant['rid'].tolist() else 0)
    df_hist['age'] = df_hist['uid'].apply(lambda x: df_user[df_user['uid'] == x]['age'].iloc[0] if x in df_user['uid'].tolist() else 0)
    df_hist['gender'] = df_hist['uid'].apply(lambda x: df_user[df_user['uid'] == x]['gender'].iloc[0] if x in df_user['uid'].tolist() else 0)
    df_hist['occupation'] = df_hist['uid'].apply(lambda x: df_user[df_user['uid'] == x]['occupation'].iloc[0] if x in df_user['uid'].tolist() else 0)

    # create avg_rating, avg_price, rcat_hist
    cur_user = df_hist['uid'][0]
    avg_rating_list = [0]
    avg_price_list = [0]
    rcat_hist_list = [[0]]
    r_cnt = 0
    price_valid_cnt = 0
    rating_valid_cnt = 0

    for i in range(1, df_hist.shape[0]):

        if cur_user == df_hist['uid'][i]:
            r_cnt = r_cnt + 1
            cur_hist = []
            avg_price = 0.0
            avg_rating = 0.0
            rcat_hist = []

            for j in range(1, r_cnt+1):

                if df_hist['price'][i-j] != 0:
                    price_valid_cnt = price_valid_cnt + 1
                if df_hist['rating'][i-j] != 0:
                    rating_valid_cnt = rating_valid_cnt + 1
                avg_price = avg_price + df_hist['price'][i-j]
                avg_rating = avg_rating + df_hist['rating'][i-j]

                if len(rcat_hist) < 15:
                    feature_len = len(df_hist['r_categories'][i-j])
                    if feature_len >= 3:
                        rcat_hist.extend(random.sample(df_hist['r_categories'][i-j], 3))
                    else:
                        for k in range(feature_len):
                            rcat_hist.append(df_hist['r_categories'][i-j][k])
                        for k in range(feature_len, 3):
                            rcat_hist.append(0)

            if price_valid_cnt > 0:
                avg_price = avg_price / float(price_valid_cnt)
            if rating_valid_cnt > 0:
                avg_rating = avg_rating / float(rating_valid_cnt)

            avg_rating_list.append(avg_rating)
            avg_price_list.append(avg_price)
            rcat_hist_list.append(rcat_hist)

        else:
            cur_user = df_hist['uid'][i]
            r_cnt = 0
            price_valid_cnt = 0
            rating_valid_cnt = 0
            avg_rating_list.append(0)
            avg_price_list.append(0)
            rcat_hist_list.append([0])
            
    df_hist['avg_price'] = avg_price_list
    df_hist['avg_rating'] = avg_rating_list
    df_hist['rcat_hist'] = rcat_hist_list

    return df_hist


def create_predict_dataset(uid, df_restaurant_in):
    """
    Create dataset for prediciton
    Args:
        uid (str): uid of user
        df_restaurant_in (df): dataframe of restaurant profiles
    Returns:
        df_user (df): dataframe of user profile
        df_restaurant_model (df): dataframe of dataset for prediction
    """
    print("create_predict_dataset")
    # extract useful columns from data
    df_restaurant = df_restaurant_in.copy()
    df_restaurant['locale_name'] = df_restaurant['alias'].apply(extract_name)
    df_restaurant['has_chinese_name'] = df_restaurant['locale_name'].apply(lambda x: True if re.search(u'[\u4e00-\u9fff]', x) else False)

    combined_addr = df_restaurant['location']
    addr_chin = []
    addr_eng = []
    for addr_list in combined_addr:
        if len(addr_list['display_address']) == 3:
            addr_eng.append(addr_list['display_address'][0])
            addr_chin.append(addr_list['display_address'][1])
        elif len(addr_list['display_address']) == 2:
            addr_eng.append(addr_list['display_address'][0])
            addr_chin.append('')
        else:
            addr_eng.append('')
            addr_chin.append('')

    df_restaurant['chinese_address'] = addr_chin
    df_restaurant['english_address'] = addr_eng

    lat_list = []
    long_list = []
    coords = df_restaurant['coordinates']
    for coord in coords:
        lat_list.append(coord['latitude'])
        long_list.append(coord['longitude'])
    df_restaurant['latitude'] = lat_list
    df_restaurant['longitude'] = long_list

    # map integer to the restaurant categories
    df_restaurant['categories'] = df_restaurant['categories'].apply(flatten_dictlist)
    df_restaurant['categories'] = df_restaurant['categories'].apply(lambda x: [feature_mapping[category] for category in x])
    df_restaurant['id'] = df_restaurant['id'].str.strip()

    df_restaurant_model = df_restaurant[['id', 'review_count', 'rating', 'price', 'categories']].copy()

    mean_price = 1.8306 # average of all restaurants, calculated from all restaurants
    mean_rating = 3.7553 # average of all restaurants

    df_restaurant_model['price'] = df_restaurant_model['price'].apply(lambda x: len(x) if not pd.isnull(x) else mean_price)
    df_restaurant_model['rating'] = df_restaurant_model['rating'].apply(lambda x: mean_rating if x == 0 or pd.isnull(x) else x)

    df_user = retrieve_user_history(uid)

    # combine dataframe to pass into the model
    df_restaurant_model[['occupation', 'gender', 'age', 'avg_rating', 'avg_price', 'rcat_hist']] = [df_user[['occupation', 'gender', 'age', 'avg_rating', 'avg_price', 'rcat_hist']].iloc[0].values for i in df_restaurant_model.index]

    # normalize data
    df_restaurant_model['rating'] = df_restaurant_model['rating'].apply(lambda x: (x-1)/(5-1))
    df_restaurant_model['price'] = df_restaurant_model['price'].apply(lambda x: (x-1)/(4-1))
    df_restaurant_model['avg_rating'] = df_restaurant_model['avg_rating'].apply(lambda x: (x-1)/(5-1))
    df_restaurant_model['avg_price'] = df_restaurant_model['avg_price'].apply(lambda x: (x-1)/(4-1))

    return df_user, df_restaurant_model

    
def retrieve_user_history(uid):
    """
    Retrieve user profile and calculate features of visited restaurants
    Args:
        uid (str): uid of user
    Returns:
        df_user (df): dataframe of user profile
    """
    df_user = get_user_profile(uid)
    df_user['avg_rating'] = [0]
    df_user['avg_price'] = [0]
    df_user['rcat_hist'] = [[0]]

    if "review" not in df_user['history'][0]:
      return df_user

    avg_rating = 0.0
    avg_price = 0.0
    valid_price_cnt = 0
    valid_rating_cnt = 0
    rcat_hist = []

    reviews = df_user['history'][0]['review']
    timestamps = list(df_user['history'][0]['review'].keys())

    # calculate avg_rating, avg_price, rcat_hist from reviews
    for timestamp in reversed(timestamps):
        rating = reviews[timestamp]['rating']
        if rating != 0.0:
            avg_rating = avg_rating + rating
            valid_rating_cnt = valid_rating_cnt + 1
        
        df_r = get_restaurant_profile(reviews[timestamp]['rid'])
        if df_r is None:
            continue
    
        if len(df_r['price'][0]) > 0:
            valid_price_cnt = valid_price_cnt + 1
            avg_price = avg_price + len(df_r['price'][0])
        
        if len(rcat_hist) < 15:
            feature_len = len(df_r['categories'][0])
            if feature_len >= 3:
                rcat_hist.extend(random.sample(df_r['categories'][0], 3))
            else:
                for k in range(feature_len):
                    rcat_hist.append(df_r['categories'][0][k])
                for k in range(feature_len, 3):
                    rcat_hist.append(0)

    if valid_rating_cnt > 0:  
        avg_rating = avg_rating / float(valid_rating_cnt)
    if valid_price_cnt > 0:
        avg_price = avg_price / float(valid_price_cnt)

    df_user['avg_rating'] = [avg_rating]
    df_user['avg_price'] = [avg_price]
    df_user['rcat_hist'] = [rcat_hist]

    return df_user


def cal_swipe_score(df_user, rid): 
    """
    Calculate swipe score
    Args:
        df_user (df): dataframe of user profile
        rid (str): restaurant id
    Returns:
        (double): swipe score
    """
    accept_score = 0.0
    reject_score = 0.0
    if 'swipe' in df_user['history'][0]:
    # for rid in df_restaurant['id']:
        if rid in df_user['history'][0]['swipe']:
            if 'acceptCount' in df_user['history'][0]['swipe'][rid]:
                accept_score = df_user['history'][0]['swipe'][rid]['acceptCount'] * 0.05
                if accept_score > 1.0:
                    accept_score = 1.0
            if 'rejectCount' in df_user['history'][0]['swipe'][rid]:
                reject_score = df_user['history'][0]['swipe'][rid]['rejectCount'] * -0.05
                if reject_score < -1.0:
                    reject_score = -1.0

    return accept_score + reject_score

def cal_pr_score(u_match_list, r_match_list):
    """
    Calculate price level and rating similarity score
    Args:
        u_match_list (df): contains the avg_rating and avg_price of user
        r_match_list (df): contains the rating and price of restaurants
    Returns:
        pr_score: price and rating similarity score
        pr_score_valid: contain the list of True/False to identify valid price and rating
    """
    pr_score = [0.0] * len(r_match_list)
    compare_price = []
    compare_rating = []
    u_price_valid = (u_match_list['avg_price'][0] != 0.0)
    u_rating_valid = (u_match_list['avg_rating'][0] != 0.0)
    pr_score_valid = [True] * len(r_match_list)

    for i in range(r_match_list.shape[0]):
        # check if the value is valid
        if u_price_valid:
            compare_price.append(r_match_list['price'][i] != 0.0)
        if u_rating_valid:
            compare_rating.append(r_match_list['rating'][i] != 0.0) 

    u_match_list['avg_rating'] = (u_match_list['avg_rating']-1)/4
    u_match_list['avg_price'] = (u_match_list['avg_price']-1)/3
    r_match_list['rating'] = (r_match_list['rating']-1)/4
    r_match_list['price'] = (r_match_list['price']-1)/3

    for i in range(r_match_list.shape[0]):
        if u_price_valid and u_rating_valid and compare_price[i] and compare_rating[i]:
            pr_score[i] = 0.5 * (1 - abs(u_match_list['avg_price'][0] - r_match_list['price'][i])) + 0.5 * (1 - abs(u_match_list['avg_rating'][0] - r_match_list['rating'][i]))
        elif u_price_valid and compare_price[i]:
            pr_score[i] = (1 - abs(u_match_list['avg_price'][0] - r_match_list['price'][i]))
        elif u_rating_valid and compare_rating[i]:
            pr_score[i] = (1 - abs(u_match_list['avg_rating'][0] - r_match_list['rating'][i]))
        else:
            pr_score_valid[i] = False
    
    return pr_score, pr_score_valid

# initialization
model = get_model()
append_eval_log(RESULT_LOCAL_PATH, ["timestamp_trained", "evaluation_loss"])

def handler(event, context):

    valid = True    # valid request
    data = event
    statusCode = 200
    data_json = {}

    if "command" in event:
        data = event
    elif "requestContext" in event:
        if event['isBase64Encoded']:
            data = json.loads(base64.b64decode(event['body']))
        else:
            data = json.loads(event['body'])
    print(data)
    if data['command'] == "predict":
        keys = ['uid', 'latitude', 'longitude']
        if all(key in data for key in keys):
            if len(data['uid']) < 35 and type(data['latitude']) == float and type(data['longitude'] == float):
                uid = data['uid']
                print('uid: ', uid)
                if 'radius' not in data:
                    data['radius'] = 1000
                if 'price' not in data:
                    data['price'] = "1,2,3,4"
                if 'yelpCategories' in data:
                    print("before cat: ", data['yelpCategories'])
                    data['yelpCategories'] = json.loads(data['yelpCategories'])
                    print("after cat: ", data['yelpCategories'])
                else:
                    data['yelpCategories'] = []
                resps = search_nearby_restaurant(data['latitude'], data['longitude'], data['radius'], data['price'], data['yelpCategories'])
                # print('resps: ', resps)
                if len(resps) > 0:
                    df_restaurant = decode_resp(resps)
                    # print('df_restaurant: ', df_restaurant)
                    df_user, df_restaurant_model = create_predict_dataset(uid, df_restaurant)
                    # df_restaurant['price'] = df_restaurant['price'].apply(lambda x: len(x) if not pd.isnull(x) else 0)
                    global model
                    prediction = model_predict(model, df_restaurant_model)

                    flat_prediction = flatten_list(prediction)
                    df_restaurant['score'] = flat_prediction
                    if 'diningPreferences' in df_user:
                        preference_score = [sum([u_preference == r_cat for u_preference in df_user['diningPreferences'][0] for r_cat in df_restaurant['categories'][i]])/len(df_user['diningPreferences'][0]) for i in range(df_restaurant.shape[0])]
                    else:
                        preference_score = 0.0
                    swipe_score = df_restaurant['id'].apply(lambda x: cal_swipe_score(df_user, x))
                    u_match_list = df_user[['avg_rating', 'avg_price']].copy()
                    r_match_list = df_restaurant[['rating', 'price']].copy()
                    r_match_list['price'] = r_match_list['price'].apply(lambda x: len(x) if not pd.isnull(x) else 0)

                    pr_score, pr_score_valid = cal_pr_score(u_match_list, r_match_list)
                    
                    sim_score = preference_score + swipe_score + pr_score
                    for i in range(len(sim_score)):
                        if pr_score_valid[i]:
                            sim_score[i] = (sim_score[i] - (-1))/3 # normalize with only preference and swipe
                        else:
                            sim_score[i] = (sim_score[i] - (-1))/4

                    df_restaurant['score'] = [df_restaurant['score'][i]*0.5 + sim_score[i]*0.5 for i in range(df_restaurant.shape[0])]        
                    df_restaurant = df_restaurant.sort_values(by=['score'], ascending=False)

                    # convert to json
                    res = {
                        'total': df_restaurant.shape[0],
                        'businesses': df_restaurant,
                        'region': resps[-1]['region']
                        }
                    data_json = json.dumps(res, default=lambda df_restaurant: json.loads(df_restaurant.to_json(orient='records')))
                
    elif data['command'] == "train":
        df_user, new_data_cnt = get_all_user_profile()
        if new_data_cnt >= 1000:
            print("Model Re-training - Start")
            timestamp_trained = get_timestamp_trained()
            df_hist = create_training_dataset(df_user, timestamp_trained)
            model = train_model(model, df_hist, timestamp_trained)
            data_json = "Model Re-training - Success"
            print(data_json)
        else:
            data_json = "Model Re-training - Not enough data"
            print(data_json)
    elif data['command'] == "get_eval":
        df_eval = pd.read_csv(RESULT_LOCAL_PATH, index_col=False)
        data_json = df_eval.to_json(orient="columns")
    else:
        valid = False
    
    if not valid:
        data_json = "POST Params missing - 'uid', 'latitude', 'longitude' "
        statusCode = 400

    response = {
        "isBase64Encoded": False,
        "statusCode": statusCode,
        "headers": {
            "content-type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": data_json,
    }

    return response