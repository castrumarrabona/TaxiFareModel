# imports
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        return pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.run().predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        #print(rmse)
        return rmse


if __name__ == "__main__":
    # get data
    #df = get_data()
    # clean data
    #df = clean_data(df)
    # set X and y
    #y = df["fare_amount"]
    #X = df.drop("fare_amount", axis=1)
    # hold out
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    #pipeline = set_pipeline
    #run(X_train, y_train, pipeline)
    # evaluate
    #rmse = evaluate(X_val, y_val, pipeline)
    print('TODO')
