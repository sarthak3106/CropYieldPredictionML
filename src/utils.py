import os 
import sys 
import pickle 
import pymongo
import pandas as pd
from exception import CustomException
from sklearn.metrics import r2_score
from dotenv import load_dotenv
from logger import logging
load_dotenv()
def save_function(file_path, obj): 
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok= True)
    with open (file_path, "wb") as file_obj: 
        pickle.dump(obj, file_obj)

def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
           
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)


def load_obj(file_path):
    try: 
        logging.info(f"file path during load: {file_path}")
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object function in utils")
        raise CustomException(e, sys)


MONGO_URI = os.environ.get('MONGO_URI')
DB_NAME = os.environ.get('DB_NAME')
COLLECTION_NAME = os.environ.get('COLLECTION_NAME')
#fetching data
def retrieve_data_from_mongodb():
    try:
        Database_url = pymongo.MongoClient(MONGO_URI)
        db = Database_url[DB_NAME]
        collection = db[COLLECTION_NAME]
        data_from_mongodb = list(collection.find())
        df = pd.DataFrame(data_from_mongodb)
        df.drop('_id', axis=1, inplace=True)
        return df

    except Exception as e:
        logging.error('Exception occurred during MongoDB data retrieval')
        raise CustomException(e,  sys)
