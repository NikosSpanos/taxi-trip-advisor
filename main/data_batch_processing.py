#!/usr/bin/env python

import polars as pl
import boto3
import sys
import logging
import configparser
import os
import glob
from datetime import datetime
sys.path.append('./src')
from custom_logger import setup_logger
from staging_modules import load_json_toDF, \
    write_df_toJSON, \
    fix_data_type, \
    remove_rows_from_future, \
    remove_negative_charges, \
    remove_equal_pickup_dropoff_times, \
    feature_engineer_trip_duration, \
    feature_engineer_trip_hour, \
    feature_engineer_trip_daytime, \
    retrieve_latest_modified_file, \
    write_df_toCSV, \
    write_df_toJSON_v2

def main(logger_object:logging.Logger):
    
    #========================================================
    # INITIALIZE MODULE VARIABLES
    #========================================================
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))
    application_path = config.get("settings", "application_path")
    samples_value = config.get("ml-settings", "samples_value")
    samples_str = config.get("ml-settings", "samples_str")
    execution_timestamp = datetime.now().strftime('%Y%m%d')

    # Establish connection to S3 bucket resource
    s3_bucket_resource = boto3.resource(
        service_name  = "s3",
        region_name = config.get("aws-settings", "aws_bucket_region"),
        aws_access_key_id = config.get("aws-settings", "aws_access_key"),
        aws_secret_access_key = config.get("aws-settings", "aws_secret_key"),
    )
    bucket_name = config.get("aws-settings", "aws_bucket_name")
    s3_bucket_object = s3_bucket_resource.Bucket(bucket_name)
    #------------------------------------------------------------------------------------

    # directory = "{0}/data/landing".format(application_path)
    # files = os.listdir(directory)
    # for file in files:
    #     # Construct the old file name
    #     old_file = os.path.join(directory, file)
    #     # Construct the new file name
    #     new_file = os.path.join(directory, file + '.json')
    #     # Rename the file
    #     os.rename(old_file, new_file)

    #========================================================
    # LOAD THE CHUNKS OF JSON FILES INTO POLARS DATAFRAME
    # == Comment: Once I have written the data into 1 single JSON file I will read that file from now on for the preprocessing part
    #========================================================
    # df = pl.DataFrame([])
    # schema = {
    #     "tpep_pickup_datetime": pl.Utf8,
    #     "tpep_dropoff_datetime": pl.Utf8,
    #     "trip_distance": pl.Utf8,
    #     "pulocationid": pl.Utf8,
    #     "dolocationid": pl.Utf8,
    #     "fare_amount": pl.Utf8,
    #     "tolls_amount": pl.Utf8
    # }
    # cols_list = list(schema.keys())
    # logger_object.info(cols_list)
    # df = load_json_toDF("{0}/data/landing/".format(application_path), df, cols_list, logger_object)
    # print(df.head())
    # write_df_toJSON("{0}/data/staging/unprocessed/{1}".format(application_path, execution_timestamp), df, "yellow_taxi_trip_unprocessed_data", logger_object)

    #========================================================
    # READ THE SINGLE JSON FILE FROM STAGING FOLDER
    #========================================================
    relative_path = "{0}/data/staging/unprocessed/{1}".format(application_path, "20231022")
    json_file = glob.glob("{0}/*.json".format(relative_path))
    df = pl.read_json(json_file[0])
    print(df.head())

    #========================================================
    # CLEANING / PREPROCESSING  RAW DATA
    #========================================================

    #=========================
    # SECTION 1: FIX DATA TYPES
    #=========================
    cast_str = pl.Utf8
    cast_categ = pl.Categorical
    cast_int = pl.Int64
    cast_float = pl.Float64
    dt_format = "%Y-%m-%dT%H:%M:%S.000"

    dtype_map = {
        "tpep_pickup_datetime": "datetime",
        "tpep_dropoff_datetime": "datetime",
        "pulocationid": cast_str,
        "dolocationid": cast_str,
        "trip_distance": cast_float,
        "fare_amount": cast_float,
        "tolls_amount": cast_float
    }
    df = fix_data_type(df, dtype_map, dt_format)

    # #==================================================
    # REMOVE ROWS NOT FOLLOWING GENERAL COLUMN RULES
    # #==================================================

    # Rows with pickup, dropoff datetimes after/before dataset year
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    dataset_year = datetime(2021,1,1)
    df = remove_rows_from_future(df, cols, dataset_year, logger_object)

    # Rows with numerical negative values (float columns)
    cols = ["fare_amount", "tolls_amount", "trip_distance"]
    df = remove_negative_charges(df, cols, logger_object)

    # Rows with equal pickup == dropoff datetimes or pickup date > dropoff date
    df = remove_equal_pickup_dropoff_times(df, "tpep_pickup_datetime", "tpep_dropoff_datetime", logger_object)

    # Compute the number of null records per column
    df_nulls = df.select(pl.all().is_null().sum()).to_dicts()[0]
    null_column_names = [k for k, v in df_nulls.items() if v > 0]
    logger_object.info("Column names with null values: {0}".format(null_column_names))

    # #==================================================
    # FEATURE ENGINEERING
    # #==================================================
    df = feature_engineer_trip_duration(df, "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_duration")
    print(df.select(pl.col("trip_duration")).head(10))
    hour_tuple = [("tpep_pickup_datetime", "pickup"), ("tpep_dropoff_datetime", "dropoff")]
    df = feature_engineer_trip_hour(df, hour_tuple)
    daytime_mapper = {"Rush-Hour": 1, "Overnight": 2, "Daytime": 3}
    daytime_tuple = [("pickup_hour", "pickup"), ("dropoff_hour", "dropoff")]
    df = feature_engineer_trip_daytime(df, daytime_mapper, daytime_tuple)
    print(df.select(pl.col("pickup_daytime")).head(10))

    # #==================================================
    # ENRICH DATA WITH NYC ZONE NAMES
    # #==================================================
    relative_path = "{0}/data/geospatial/".format(application_path)
    df_geo = pl.read_json(retrieve_latest_modified_file(relative_path, False))
    print(df_geo.head())

    merged_data = pl.DataFrame([])
    geo_cols = ["objectid", "zone"]

    # Merge on PULocationID
    merged_data = df.join(df_geo.select(geo_cols), left_on=pl.col("pulocationid"), right_on=pl.col("objectid"), how='left')
    merged_data = merged_data.rename({"zone": "puzone"})

    # Merge on DOLocationID
    merged_data = merged_data.join(df_geo.select(geo_cols), left_on=pl.col("dolocationid"), right_on=pl.col("objectid"), how='left')
    merged_data = merged_data.rename({"zone": "dozone"})
    print(merged_data.columns)
    print(list(zip(merged_data.dtypes, merged_data.columns)))

    # Re-Compute the number of null records per column
    df_nulls = merged_data.select(pl.all().is_null().sum()).to_dicts()[0]
    null_column_names = [k for k, v in df_nulls.items() if v > 0]
    logger_object.info("Column names with null values: {0}".format(null_column_names))

    # #==================================================
    # WRITE PROCESSED TABLE TO JSON
    # #==================================================
    write_df_toJSON("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data, "yellow_taxi_trip_processed_data", logger_object)
    write_df_toCSV("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data, "yellow_taxi_trip_processed_data", logger_object)

    # Shuffle the dataframe and filter top 200_000 rows
    merged_data_shuffled = merged_data.select(pl.col("*").shuffle(123))
    merged_data_shuffled = merged_data_shuffled.head(samples_value)
    write_df_toJSON("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data_shuffled, "yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)
    write_df_toCSV("{0}/data/staging/processed/{1}".format(application_path, execution_timestamp), merged_data_shuffled, "yellow_taxi_trip_processed_data_{0}".format(samples_str), logger_object)

    logger_object.info("ENRICHMENT COMPLETED - Geospatial data imported to dataframe.")

if __name__ == "__main__":
    log_filename = f"batch_processing/batch_processing_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(log_filename)
    try:
        main(logger)
        logger.info("SUCCESS: Batch processing/cleaning/feature-engineering completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: Batch processing/cleaning/feature-engineering failed.")