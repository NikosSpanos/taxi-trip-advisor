import polars as pl
import numpy as np
import xgboost as xgb
import sys
import logging
import configparser
import os
import mlflow
import mlflow.sklearn
import setuptools
import distutils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
sys.path.append('./src')
from custom_logger import setup_logger
from staging_modules import retrieve_latest_modified_file, create_folder, write_df_toJSON, write_df_toCSV
from ml_modules import remove_null_values, \
    label_encode_column, \
    train_linear_regressor, \
    train_randomforest_regressor, \
    train_xgboost_regressor, \
    make_predictions, \
    save_label_encoder

def duration_predictor(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))
    application_path = config.get("settings", "application_path")
    processed_data_dt = config.get("ml-settings", "processed_dt")
    samples_str = config.get("ml-settings", "samples_str")
    ml_model_name = config.get("ml-settings", "duration_model_name")
    split_perce:float = float(config.get("ml-settings", "train_test_split_perce"))
    relative_path:str = "{0}/data/staging/processed/{1}".format(application_path, processed_data_dt)
    execution_timestamp:datetime = datetime.now().strftime('%Y%m%d')
    artifact_path:str = os.path.join(application_path, "model_artifacts", execution_timestamp)
    create_folder(artifact_path)
    RANDOM_SEED:int = 42
    np.random.seed(RANDOM_SEED)
    
    #========================================================
    # READ THE PROCESSED-DATA JSON FILE FROM STAGING FOLDER
    #========================================================
    df = pl.read_json(retrieve_latest_modified_file(relative_path, True, samples_str))
    print(df.shape)

    #========================================================
    # COMPUTE NULL VALUES AND REMOVE THEM
    #========================================================
    # print(df.select(pl.col(["pulocationid", "dolocationid", "puzone", "dozone"])).filter(pl.col("puzone").is_null())) 
    # print(df.select(pl.col(["pulocationid", "dolocationid", "puzone", "dozone"])).filter(pl.col("dozone").is_null()))
    df_nulls = df.select(pl.col(["puzone", "dozone"]).is_null().sum()).to_dicts()[0]
    null_column_names = [(k,v) for k, v in df_nulls.items() if v > 0]
    logger_object.info("Columns with null values (col, value): {0}".format(null_column_names))
    remove_null_from = ["puzone", "dozone"]
    df = remove_null_values(df, remove_null_from)
    print(df.shape)
    write_df_toJSON("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), df, "nulls_pruned_yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)
    write_df_toCSV("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), df, "nulls_pruned_yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)

    #========================================================
    # POLARS TO PANDAS FOR BETTER HANDLING FROM SKLEARN/XGBOOST
    #========================================================
    df = df.to_pandas()

    #========================================================
    # LABEL ENCODE CATEGORICAL VARIABLES
    #========================================================
    label_encoder = LabelEncoder()
    encode_cols = ["puzone", "dozone"]
    fit_encoder = True
    for name in encode_cols:
        df, label_encoder, refit = label_encode_column(df, name, label_encoder, fit_encoder, logger_object)
        if refit:
            save_label_encoder(label_encoder, artifact_path + "/{0}_label_encoder.joblib".format(name))

    #========================================================
    # ISOLATE X, Y FEATURES AND SPLIT THEM TO TRAIN/TEST SAMPLES
    #========================================================
    # x_features = ["puzone_encoded", "dozone_encoded", "trip_distance", "pickup_daytime"]
    x_features = ["puzone_encoded", "dozone_encoded", "pickup_daytime"]
    y_features = ["trip_duration"]
    X = df[x_features]
    y = df[y_features]
    logger_object.info(y.describe())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_perce, random_state=RANDOM_SEED)

    custom_mlruns_path:str = os.path.join(application_path, "mlruns", execution_timestamp)
    mlflow.set_tracking_uri("file://{0}".format(custom_mlruns_path))
    mlflow.set_experiment("trip-{0}-prediction-model".format(ml_model_name))

    model_regressors = ["xgboost", "linear", "randomforest"]
    best_score:np.float64 = np.inf
    for model_name in model_regressors:
        if model_name == "xgboost":
            with mlflow.start_run(run_name="{0}-model".format(model_name), nested=False):
                logger_object.info("Strarted training/evaluating {0} regressor".format(model_name))
                dtrain = xgb.DMatrix(X_train, label=y_train)
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "learning_rate": 0.01,
                    "max_depth": 10,
                    "num_parallel_tree": 100
                }
                model = train_xgboost_regressor(params, dtrain)
                dtest = xgb.DMatrix(X_test)
                mlflow.log_params(params)
                rmse, second_metric_value, second_metric_name = make_predictions(model_name, model, dtest, y_test, logger_object)
                mlflow.log_metric("root-mean-squared-error", rmse)
                mlflow.log_metric("{0}".format(second_metric_name), second_metric_value)
                logger_object.info("Completed training/evaluating {0} regressor".format(model_name))
                if rmse < best_score:
                    logger_object.info("Found a model that improved RMSE from {0} to {1}".format(best_score, rmse))
                    best_score = rmse
                    mlflow.sklearn.log_model(model, "best_{0}_recommendation".format(ml_model_name), serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
                logger_object.info("========================================================================================")
            mlflow.end_run()
        elif model_name == "linear":
            with mlflow.start_run(run_name="{0}-model".format(model_name), nested=False):
                logger_object.info("Strarted training/evaluating {0} regressor".format(model_name))
                dtrain = X_train.values
                dtest = X_test.values
                params = {
                    "fit_intercept": True,
                    "copy_X": True
                }
                model = train_linear_regressor(dtrain, y_train.values, params)
                mlflow.log_params(params)
                rmse, second_metric_value, second_metric_name = make_predictions(model_name, model, dtest, y_test, logger_object)
                mlflow.log_metric("root-mean-squared-error", rmse)
                mlflow.log_metric("{0}".format(second_metric_name), second_metric_value)
                logger_object.info("Completed training/evaluating {0} regressor".format(model_name))
                if rmse < best_score:
                    logger_object.info("Found a model that improved RMSE from {0} to {1}".format(best_score, rmse))
                    best_score = rmse
                    mlflow.sklearn.log_model(model, "best_{0}_recommendation".format(ml_model_name), serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
                logger_object.info("========================================================================================")
            mlflow.end_run()
        else:
            with mlflow.start_run(run_name="{0}-model".format(model_name), nested=False):
                logger_object.info("Strarted training/evaluating {0} regressor".format(model_name))
                dtrain = X_train.values
                dtest = X_test.values
                params = {
                    "n_estimators": 100,
                    "criterion": "squared_error",
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": RANDOM_SEED
                }
                model = train_randomforest_regressor(dtrain, y_train.values.ravel(), params)
                mlflow.log_params(params)
                rmse, second_metric_value, second_metric_name = make_predictions(model_name, model, dtest, y_test, logger_object)
                mlflow.log_metric("root-mean-squared-error", rmse)
                mlflow.log_metric("{0}".format(second_metric_name), second_metric_value)
                logger_object.info("Completed training/evaluating {0} regressor".format(model_name))
                if rmse < best_score:
                    logger_object.info("Found a model that improved RMSE from {0} to {1}".format(best_score, rmse))
                    best_score = rmse
                    mlflow.sklearn.log_model(model, "best_{0}_recommendation".format(ml_model_name), serialization_format = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
                logger_object.info("========================================================================================")
            mlflow.end_run()
    logger_object.info("Completed training/evaluating {0}-model".format(ml_model_name))

if __name__ == "__main__":
    log_filename = "trip_duration_model/duration_recommendation_model_logs_{0}.txt".format(datetime.now().strftime('%Y_%m_%d_%H_%M'))
    logger = setup_logger(log_filename)
    try:
        duration_predictor(logger)
        logger.info("SUCCESS: duration recommendation model completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: duration recommendation model failed.")