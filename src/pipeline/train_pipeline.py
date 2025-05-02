import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(data_path):
    """
    Create a sample dataset for demonstration purposes
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Create a sample dataset with the required columns
    np.random.seed(42)
    n_samples = 500
    
    # Sample data
    data = {
        'N': np.random.uniform(0, 140, n_samples),
        'P': np.random.uniform(5, 145, n_samples),
        'K': np.random.uniform(5, 205, n_samples),
        'pH': np.random.uniform(3.5, 10, n_samples),
        'rainfall': np.random.uniform(20, 300, n_samples),
        'temperature': np.random.uniform(8, 45, n_samples),
        'Area_in_hectares': np.random.uniform(0.1, 10, n_samples),
        'State_Name': np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Uttar Pradesh'], n_samples),
        'Crop_Type': np.random.choice(['Kharif', 'Rabi', 'Zaid'], n_samples),
        'Crop': np.random.choice(['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'], n_samples),
        'Production': np.random.uniform(0.5, 50, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save the dataset
    df.to_csv(data_path, index=False)
    logger.info(f"Sample dataset created and saved to {data_path}")
    return df

def run_training_pipeline():
    try:
        # Create archive directory if it doesn't exist
        os.makedirs('archive', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Data paths
        import os

# Construct the absolute path to the preprocessor file
base_dir = os.path.abspath(os.path.dirname("data/crop_data.csv"))
preprocessor_path = os.path.join(base_dir, 'archive', 'preprocessor.pkl')

# Load the preprocessor
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

        
        # Create or load dataset
        if not os.path.exists(data_path):
            logger.info("Creating sample dataset")
            df = create_sample_data(data_path)
        else:
            logger.info(f"Loading dataset from {data_path}")
            df = pd.read_csv(data_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        
        # Define features and target
        X = df.drop('Production', axis=1)
        y = df['Production']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Define preprocessor
        numerical_columns = ['N', 'P', 'K', 'pH', 'rainfall', 'temperature', 'Area_in_hectares']
        categorical_columns = ['State_Name', 'Crop_Type', 'Crop']
        
        # Create numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        
        # Create categorical pipeline
        cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]
        )
        
        # Combine them with ColumnTransformer
        preprocessor = ColumnTransformer(
            [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ]
        )
        
        logger.info("Fitting preprocessor")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train model
        logger.info("Training Random Forest model")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_processed, y_train)
        
        # Save preprocessor and model
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate model
        train_score = model.score(X_train_processed, y_train)
        test_score = model.score(X_test_processed, y_test)
        
        print("=" * 50)
        print("Training completed successfully!")
        print(f"Train R2 Score: {train_score:.4f}")
        print(f"Test R2 Score: {test_score:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Preprocessor saved to: {preprocessor_path}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        print(f"Error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    run_training_pipeline()