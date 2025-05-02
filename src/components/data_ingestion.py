import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import logging
from exception import CustomException
from utils import retrieve_data_from_mongodb
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    Dataset_folder: str = 'dataset'
    train_data_filename: str = 'train.csv'
    test_data_filename: str = 'test.csv'
    raw_data_folder: str = 'database/data'
    raw_data_filename: str = 'crop_yield.csv'

    @property
    def train_data_path(self):
        return os.path.join(self.Dataset_folder,  self.train_data_filename)

    @property
    def test_data_path(self):
        return os.path.join(self.Dataset_folder, self.test_data_filename)

    @property
    def raw_data_path(self):
        return os.path.join(self.raw_data_folder, self.raw_data_filename)


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
#fetching data
    def retrieve_data_from_mongodb_and_save_to_csv(self):
        try:
            df=retrieve_data_from_mongodb()
           

            
            os.makedirs(self.ingestion_config.raw_data_folder, exist_ok=True)

          
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error('Exception occurred during MongoDB data retrieval and CSV creation')
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            csv_file_path = self.retrieve_data_from_mongodb_and_save_to_csv()
            logging.info('Dataset retrieved from MongoDB and saved as CSV')

            df = pd.read_csv(csv_file_path)
            logging.info('Dataset read as pandas DataFrame')

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            os.makedirs(self.ingestion_config.Dataset_folder, exist_ok=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Train and test split completed')
            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error('Exception occurred during data ingestion')
            raise CustomException(e, sys)
