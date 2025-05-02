import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictPipeline:
    def __init__(self):
        # Use the same paths as in app.py
        self.preprocessor_path = os.path.join('archive', 'preprocessor.pkl')
        self.model_path = os.path.join('archive', 'model.pkl')
        
        # Check if files exist
        if not os.path.exists(self.preprocessor_path):
            logger.error(f"Preprocessor file not found at {self.preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}")
        
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        logger.info("Prediction pipeline initialized")

    def predict(self, features):
        try:
            logger.info("Starting prediction")
            
            # Load preprocessor and model
            with open(self.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
                
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
                
            logger.info("Loaded preprocessor and model")
            
            # Transform the features
            data_scaled = preprocessor.transform(features)
            logger.info("Features transformed")
            
            # Make prediction
            prediction = model.predict(data_scaled)
            logger.info(f"Prediction: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise e


class CustomData:
    def __init__(
        self,
        N: float,
        P: float,
        K: float,
        pH: float,
        rainfall: float,
        temperature: float,
        Area_in_hectares: float,
        State_Name: str,
        Crop_Type: str,
        Crop: str
    ):
        self.N = N
        self.P = P
        self.K = K
        self.pH = pH
        self.rainfall = rainfall
        self.temperature = temperature
        self.Area_in_hectares = Area_in_hectares
        self.State_Name = State_Name
        self.Crop_Type = Crop_Type
        self.Crop = Crop
        
        logger.info("CustomData object created")

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "N": [self.N],
                "P": [self.P],
                "K": [self.K],
                "pH": [self.pH],
                "rainfall": [self.rainfall],
                "temperature": [self.temperature],
                "Area_in_hectares": [self.Area_in_hectares],
                "State_Name": [self.State_Name],
                "Crop_Type": [self.Crop_Type],
                "Crop": [self.Crop]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logger.info(f"DataFrame created with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {str(e)}")
            raise e