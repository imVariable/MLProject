import os 
import sys 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_object 
import pandas as pd 
import numpy as np 

class PredictPipeline:
    def __init__(self):
        pass 
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            preds = model.predict(data_scaled)
            return preds 
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self,
                 writing_score: float,
                 reading_score: float,
                 math_score: float,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str):
        self.writing_score = writing_score
        self.reading_score = reading_score
        self.math_score = math_score
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "writing score": [self.writing_score],
                "reading score": [self.reading_score],
                "math score": [self.math_score],
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df 
        except Exception as e:
            raise CustomException(e, sys)
        