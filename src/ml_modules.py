#!/usr/bin/env python
import polars as pl
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
import joblib
import setuptools
import distutils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def remove_null_values(df:pl.DataFrame, cols:list) -> pl.DataFrame:
    for col in cols:
        df = df.filter(~pl.col(col).is_null())
    return df

def label_encode_column(df:pd.DataFrame, cols_name:str, encoder:LabelEncoder, fit_encoder:bool, logger_obj:logging.Logger) -> (pd.DataFrame, LabelEncoder, bool):
    refit_encoder:bool = False
    # zones_list = []
    # if model_name != "duration":
    #     for col in cols:
    #         zones_list.extend(list(df[col].unique()))
    #     zones_list = set(zones_list)
    #     for zone in zones_list:
    #         if zone in list(encoder.classes_):
    #             refit_encoder = False
    # for col in cols:
    #     if refit_encoder:
    #         logger_obj.info("Refitting label encoder for column {0}.".format(col))
    #         df["{0}_encoded".format(col)] = encoder.fit_transform(df[col])
    #     else:
    #         logger_obj.info("Using already fitted label encoder for column {0}.".format(col))
    #         df["{0}_encoded".format(col)] = encoder.transform(df[col])
    # return (df, encoder, refit_encoder)
    if fit_encoder:
        logger_obj.info("Fitting label encoder for column {0}.".format(cols_name))
        df["{0}_encoded".format(cols_name)] = encoder.fit_transform(df[cols_name])
        refit_encoder = True
    else:
        logger_obj.info("Using already fitted label encoder for column {0}.".format(cols_name))
        df["{0}_encoded".format(cols_name)] = encoder.transform(df[cols_name])
    return (df, encoder, refit_encoder)

def save_label_encoder(lbl_encoder:LabelEncoder, pathname:str):
    joblib.dump(lbl_encoder, pathname)

def load_label_encoder(pathname:str) -> LabelEncoder:
    lbl_encoder = joblib.load(pathname)
    return lbl_encoder

def save_model_regressor(model, filename:str):
    model_filename = "linear_regression_model.joblib"
    joblib.dump(model, filename)

def train_xgboost_regressor(params, dtrain) -> xgb.Booster:
    model = xgb.train(params, dtrain)
    return model

def train_linear_regressor(train_x, train_y, params):
    model = LinearRegression(**params)
    model.fit(train_x, train_y)
    return model

def train_randomforest_regressor(train_x, train_y, params):
    model = RandomForestRegressor(**params)
    model.fit(train_x, train_y)
    return model

def make_predictions(model_name, model, dtest, y_test, logger_obj:logging.Logger) -> (float, float, str):
    if model_name == "linear":
        y_test = y_test.values
    elif model_name == "randomforest":
        y_test = y_test.values.ravel()
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if np.any(y_pred < 0):
        # mape = mean_absolute_percentage_error(y_test, y_pred)
        mape = mean_absolute_error(y_test, y_pred)
        metric_selected_value = mape
        # metric_selected_name = "mean-absolute-percentage-error"
        metric_selected_name = "mean-absolute-error"
    else:
        msle = mean_squared_log_error(y_test, y_pred)
        metric_selected_value = msle
        metric_selected_name = "mean-squared-logarithmic-error"
    logger_obj.info("root-mean-squared-error: %f" % (rmse))
    logger_obj.info("%s: %f" % (metric_selected_name, metric_selected_value))
    return (rmse, metric_selected_value, metric_selected_name)