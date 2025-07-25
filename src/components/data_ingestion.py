import os 
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.exception import CustomException 
from src.logger import logging 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 
from src.components.data_transform import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass 
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv') 
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() 
        
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('/Users/m/Documents/MLProject/notebook/data/StudentsPerformance.csv')
            logging.info('Data read as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test split already initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed !!!")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e: 
            raise CustomException(e, sys)
        
        
"""
    Testing Data Ingestion file to create artifacts and logs
"""        
        
if __name__ == '__main__':
    # obj = DataIngestion() 
    # obj.initiate_data_ingestion()
    
    # obj = DataIngestion() 
    # train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    # data_transformation = DataTransformation() 
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    
    obj = DataIngestion() 
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()  
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer() 
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    

