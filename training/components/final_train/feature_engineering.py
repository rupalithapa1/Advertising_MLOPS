import os
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from training.custom_logging import info_logger,error_logger
from training.exception import FeatureEngineeringError,handle_exception

from training.entity.config_entity import FeatureEngineeringConfig
from training.configuration_manager.configuration import ConfigurationManager


class FeatureEngineering:
    def __init__(self,config: FeatureEngineeringConfig):
        self.config = config

    def load_saved_data(self):
        try:

            info_logger.info("loading final training saved data for transformation")

            final_train_data_path = os.path.join(self.config.final_train_data_path,'Train.npz')
            final_test_data_path = os.path.join(self.config.final_test_data_path,'Test.npz')

            final_train_data = np.load(final_train_data_path,allow_pickle=True)
            final_test_data = np.load(final_test_data_path,allow_pickle=True)

            xtrain = final_train_data["xtrain"]
            xtest = final_test_data["xtest"]
            ytrain = final_train_data["ytrain"]
            ytest = final_test_data["ytest"]

            info_logger.info("final training data saved completed")

            return xtrain,xtest,ytrain,ytest
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

    def transform_data(self,xtrain,xtest,ytrain,ytest):
        try:
            info_logger.info("transforming final training data")

            transform_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')), #missing value imputation
                ('scaler',StandardScaler())     #standardization
            ])

            transform_pipeline.fit(xtrain)

            #saving the feature_transformer pipeline as artifacts
            #save the pipeline
        
            pipeline_path =  os.path.join(self.config.root_dir,"pipeline.joblib")
            joblib.dump(transform_pipeline,pipeline_path)

            #transforming xtrain and xtest
            xtrain= transform_pipeline.transform(xtrain)
            xtest = transform_pipeline.transform(xtest)

            info_logger.info("transformed final training data")

            return xtrain,xtest,ytrain,ytest
        
        except Exception as e:
            handle_exception (e, FeatureEngineeringError)


    def save_transformed_data(self,xtrain,xtest,ytrain,ytest):
        try:
            info_logger.info("Saving final training transformed data")

            final_transform_data_path = self.config.root_dir

            #save xtrain and ytrain to train.npz
            #save xtest and ytest to test.npz
            np.savez(os.path.join(final_transform_data_path,'Train.npz'),xtrain=xtrain,ytrain=ytrain)
            np.savez(os.path.join(final_transform_data_path,'Test.npz'),xtest=xtest,ytest=ytest)
            
            info_logger.info("final training transformed data saved successfully")

            with open(self.config.STATUS_FILE, "w") as f:
             f.write(f"Feature Engineering status: True")

        except Exception as e:
            handle_exception(e,FeatureEngineeringError)   



if __name__ =="__main__":
    config = ConfigurationManager()
    feature_engineering_config = config.get_feature_engineering_config()

    feature_engineering = FeatureEngineering(config = feature_engineering_config)
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()
    xtrain, xtest, ytrain, ytest = feature_engineering.transform_data( xtrain, xtest, ytrain, ytest)
    feature_engineering.save_transformed_data( xtrain, xtest, ytrain, ytest)

