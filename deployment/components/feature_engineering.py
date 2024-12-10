from deployment.custom_logging import info_logger,error_logger
from deployment.exception import FeatureEngineeringError
from deployment.exception import handle_exception

import sys
import os
from pathlib import Path
import joblib
import numpy as np

class FeatureEngineering:
    def __init__(self):
        pass

    def transform_data(self,data):
        try:
            transformation_pipeline_path="artifacts/feature_engineering/pipeline.joblib"
            transformation_pipeline = joblib.load(transformation_pipeline_path)

            transformed_data = transformation_pipeline.transform(data)

            return transformed_data
        except Exception as e:
            handle_exception(e,FeatureEngineeringError)

    
if __name__=="__main__":
    
    #checking whether this method is working or not by giving some data into this method
    data = np.array([[122,34.5,89]])
    feature_engineering = FeatureEngineering()
    transformed_data = feature_engineering.transform_data(data)
    print(transformed_data)