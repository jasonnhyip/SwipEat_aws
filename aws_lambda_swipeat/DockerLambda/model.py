from csv import writer
import tensorflow as tf
import boto3
import datetime
import os
import numpy as np

AWS_REGION = "ap-east-1"
KEY = ""
SECRET_KEY = ""
S3_BUCKET = ""
MODEL_DIR = "/tmp"
CURRENT_MODEL_LOCAL_PATH = "/tmp/model.h5"
RESULT_LOCAL_PATH = "/tmp/model_eval.csv"


def append_eval_log(file_name, result):
    """
    Append current timestamp and model loss to csv
    """
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(result)


def get_model():
    """
    Gets model from /tmp or load model from S3, then save to /tmp
    """
    print("get_model")
    if not os.path.exists(CURRENT_MODEL_LOCAL_PATH):

        # download model file from S3
        s3 = boto3.client("s3", aws_access_key_id=KEY, aws_secret_access_key=SECRET_KEY)
        s3.download_file(
            S3_BUCKET,
            "model.h5",
            CURRENT_MODEL_LOCAL_PATH
        )

    model = tf.keras.models.load_model(CURRENT_MODEL_LOCAL_PATH)

    return model


def save_model(model):
    """
    Upload current model to S3 as backup, save the new model in /tmp and then upload it to S3
    """
    s3 = boto3.client("s3", aws_access_key_id = KEY, aws_secret_access_key = SECRET_KEY, region_name = AWS_REGION)
    # re-upload old model as backup
    s3.upload_file(CURRENT_MODEL_LOCAL_PATH, S3_BUCKET, "model_" + datetime.now().strftime("%d%b%Y") + ".h5")
    # save the newly trained model in /tmp
    model.save(CURRENT_MODEL_LOCAL_PATH)
    # save the newly trained model in S3
    s3.upload_file(CURRENT_MODEL_LOCAL_PATH, S3_BUCKET, "model.h5")


def train_model(model, df_hist, timestamp_trained):
    """
    Convert some features in dataframe to tensor and sequence,
    feed the dataset into the model.
    Save newly re-trained model and append the evalutaion(loss) to csv.

    Args:
        model: loaded from tf.keras.models.load_model()
        df_hist: dataframe for training
        timestamp_trained: timestamp of last model re-training
    """
    if df_hist is None:
        print("Error - Fail to retrieve all user history")
        return

    df_train_x = df_hist[['rating', 'rid', 'overall_rating', 'price', 'r_categories', 'avg_rating', 'avg_price', 'rcat_hist','age', 'gender', 'occupation']].sample(frac=0.8, random_state=200)
    train_y = tf.convert_to_tensor(df_train_x['rating'], dtype=float)
    train_hist_features = tf.keras.preprocessing.sequence.pad_sequences(df_train_x['rcat_hist'], maxlen=15, padding='post')
    train_features = tf.keras.preprocessing.sequence.pad_sequences(df_train_x['r_categories'], maxlen=3, padding='post')

    df_test_x = df_hist[['rating', 'rid', 'overall_rating', 'price', 'r_categories', 'avg_rating', 'avg_price', 'rcat_hist','age', 'gender', 'occupation']].drop(df_train_x.index)
    test_y = tf.convert_to_tensor(df_test_x['rating'], dtype=float)
    test_hist_features = tf.keras.preprocessing.sequence.pad_sequences(df_test_x['rcat_hist'], maxlen=15, padding='post')
    test_features = tf.keras.preprocessing.sequence.pad_sequences(df_test_x['r_categories'], maxlen=3, padding='post')

    history = model.fit([df_train_x['overall_rating'], 
                     df_train_x['price'], 
                     train_features, 
                     df_train_x['avg_rating'],
                     df_train_x['avg_price'], 
                     train_hist_features,
                     df_train_x['gender'],
                     df_train_x['occupation']], 
                    train_y, steps_per_epoch=10, epochs=100)
    
    save_model(model)

    evaluation = model.evaluate([df_test_x['overall_rating'], 
                     df_test_x['price'], 
                     test_features, 
                     df_test_x['avg_rating'],
                     df_test_x['avg_price'], 
                     test_hist_features,
                     df_test_x['gender'],
                     df_test_x['occupation']], 
                     test_y)

    print("Model Re-training - Evaluation: ", evaluation)
    append_eval_log(RESULT_LOCAL_PATH, [timestamp_trained, evaluation])
    return model

def model_predict(model, df_restaurant_model_predict):
    """
    Feed dataset into the model for restaurant rating prediction
    Args:
        model: loaded from tf.keras.models.load_model()
        df_restaurant_model_predict: dataframe for prediction
    """

    test_features = tf.keras.preprocessing.sequence.pad_sequences(df_restaurant_model_predict['categories'], maxlen=3, padding='post')
    test_hist_features = tf.keras.preprocessing.sequence.pad_sequences(df_restaurant_model_predict['rcat_hist'], maxlen=15, padding='post')

    age = np.asarray(df_restaurant_model_predict['age']).astype('float32')
    gender = np.asarray(df_restaurant_model_predict['gender']).astype('float32')
    occupation = np.asarray(df_restaurant_model_predict['occupation']).astype('float32')

    prediction = model.predict([df_restaurant_model_predict['rating'], df_restaurant_model_predict['price'], 
                    test_features, df_restaurant_model_predict['avg_rating'], 
                    df_restaurant_model_predict['avg_price'], test_hist_features,
                    gender, occupation])

    return prediction
    

