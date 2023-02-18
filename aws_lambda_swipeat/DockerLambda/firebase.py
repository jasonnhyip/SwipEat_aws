import pyrebase
import pandas as pd
from datetime import datetime, timedelta

firebaseConfig = {
  'apiKey': "",
  'authDomain': "",
  'databaseURL': "",
  'projectId': "",
  'storageBucket': "",
  'messagingSenderId': "",
  'appId': "",
  'measurementId': ""
}

firebase = pyrebase.initialize_app(firebaseConfig) # initialize the firebase connection
db = firebase.database()

def get_user_profile(uid):
    """
    Get the user profile from firebase with target uid
    """
    user_db = db.child('users').child(uid).get() # app

    if user_db.val() and 'userData' in user_db.val():
        df_one_user = pd.DataFrame(columns = list(user_db.val()['userData']))
        df_one_user.loc[0] = list(user_db.val()['userData'].values())
        if 'history' in user_db.val():
            df_one_user['history'] =  [user_db.val()['history']]
        else:
            df_one_user['history'] = [{}];

        df_one_user['uid'] = uid
        return df_one_user


def get_restaurant_profile(rid):
    """
    Get the restaurant profile with target rid
    """
    restaurant_db = db.child("restaurant").child(rid).get()
    if restaurant_db.val():
        df_one_r = pd.DataFrame(columns=list(restaurant_db.val()))
        df_one_r.loc[0] = list(restaurant_db.val().values())
        df_one_r['rid'] = rid
        return df_one_r


def get_all_user_profile():
    """
    Get all user profiles
    """
    all_users = db.child("users").get()
    if all_users.val() is not None:

        # calculate total new data count
        data_cnt = 0
        uid_list = list(all_users.val().keys())
        for uid in uid_list:
            if 'model' in all_users.val()[uid] and 'newDataCount' in all_users.val()[uid]['model']:
                data_cnt = data_cnt + all_users.val()[uid]['model']['newDataCount']
        
        if data_cnt > 1000:
            reset_data_count(uid_list)

        users_dict = [all_users.val()[key]['userData'] for key in all_users.val().keys()]
        

        for i in range(len(users_dict)):
            users_dict[i]['uid'] = uid_list[i]
            if 'history' in all_users.val()[uid_list[i]]:
                users_dict[i]['history'] = all_users.val()[uid_list[i]]['history']
            else: 
                users_dict[i]['history'] = {}

        df_user = pd.DataFrame(users_dict)
        
        return df_user, data_cnt
    



def set_new_timestamp_trained():
    """
    Set the current timestamp as the timestamp_trained in firebase
    """
    db.child("model").child("timestamp_trained").set(int(datetime.timestamp(datetime.now())))


def reset_data_count(uid_list):
    """
    Reset newDataCount to 0 after successful model re-training
    """
    for uid in uid_list:
        db.child("users").child(uid).child("model").update({'newDataCount': 0})


def get_timestamp_trained():
    """
    Get timestamp of last re-training 
    """
    timestamp_trained_db = db.child("model").child("timestamp_trained").get().val()
    if timestamp_trained_db is None:
        one_month_ago = datetime.now() - timedelta(days=30)
        timestamp_trained = int(datetime.timestamp(one_month_ago))
    else:
        timestamp_trained = int(timestamp_trained_db)
    
    set_new_timestamp_trained();

    return timestamp_trained

