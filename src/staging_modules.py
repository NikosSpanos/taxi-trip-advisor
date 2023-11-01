#!/usr/bin/env python
import hashlib
import glob
import polars as pl
import os
import logging
import json
from datetime import datetime, timedelta

def md5_hashing(value):
    return hashlib.md5(value.encode()).hexdigest()

def get_latest_file(s3_bucket_obj, bucket_path):
    files = sorted(s3_bucket_obj.objects.filter(Prefix=bucket_path), key=lambda obj: obj.last_modified)
    latest_file = files[-1]
    return latest_file

def daytime_value(hour_value):
    if (hour_value in range(7,11)) or (hour_value in range(16,20)):
        return "Rush-Hour"
    elif hour_value in [20,21,22,23,0,1,2,3,4,5,6]:
        return "Overnight"
    else:
        return "Daytime"

def create_folder(folder_path:str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder '{0}' has been created.".format(folder_path))

def load_json_toDF(json_path:str, df, cols_list:list, logger_obj:logging.Logger):
    json_files = glob.glob("{0}*.json".format(json_path))
    dataframes = []
    for index, file in enumerate(json_files):
        logger_obj.info("{0}-".format(index) + file)
        load_df = pl.read_json(file)
        if load_df.shape[1] == 18: #exclude the dataframes with a number of cilumns < 18 (18 are the number of columns in the original dataset)
            dataframes.append(load_df.select(cols_list))
    df = pl.concat(dataframes)
    return df

def write_df_toJSON(relative_path:str, df:pl.DataFrame, filename:str, logger_obj:logging.Logger):
    create_folder(relative_path)
    filename = os.path.join(relative_path, "{0}.json".format(filename))
    df.write_json(filename)
    logger_obj.info(f"DataFrame saved as JSON under path: {filename}")

def write_df_toJSON_v2(relative_path:str, df:pl.DataFrame, filename:str, logger_obj:logging.Logger):
    create_folder(relative_path)
    filename = os.path.join(relative_path, "{0}.json".format(filename))
    with open(filename, "w") as f:
        json.dump(df, f)
    logger_obj.info(f"DataFrame saved as JSON under path: {filename}")

def write_df_toCSV(relative_path:str, df:pl.DataFrame, filename:str, logger_obj:logging.Logger):
    create_folder(relative_path)
    filename = os.path.join(relative_path, "{0}.csv".format(filename))
    df.write_csv(filename, has_header=True, separator=",")
    logger_obj.info(f"DataFrame saved as CSV under path: {filename}")

def fix_data_type(df:pl.DataFrame, type_mapping:dict, dt_format:str = None) -> pl.DataFrame:
    for column, dtype in type_mapping.items():
        if dtype == "datetime":
            df = df.with_columns(pl.col(column).str.to_datetime(dt_format))
        else:
            df = df.with_columns(pl.col(column).cast(dtype))
    return df

def remove_rows_from_future(df:pl.DataFrame, cols:list, dataset_year:datetime, logger_obj:logging.Logger) -> pl.DataFrame:
    for col in cols:
        above_current_fyear = df.filter(pl.col(col).dt.year().gt(dataset_year.year))
        before_current_fyear = df.filter(pl.col(col).dt.year().lt(dataset_year.year))
        logger_obj.info("{0} dates after dataset year: {1}".format(col, above_current_fyear.height))
        logger_obj.info("{0} dates before dataset year: {1}".format(col, before_current_fyear.height))
        # Remove rows with year greater/lower than the dataset year
        df = df.filter(
            (pl.col(col).dt.year().le(dataset_year.year)) & (pl.col(col).dt.year().gt( (dataset_year - timedelta(days=366)).year ))
        )
    return df

def remove_negative_charges(df:pl.DataFrame, cols:list, logger_obj:logging.Logger) -> pl.DataFrame:
    for col in cols:
        if col == "tolls_amount":
            negative_charges = df.filter(pl.col(col).lt(0))
            logger_obj.info("{0} with negative values (<0): {1}".format(col, negative_charges.height))
            df = df.filter(pl.col(col).ge(0))
        else:
            negative_charges = df.filter(pl.col(col).le(0))
            logger_obj.info("{0} with negative values (<=0): {1}".format(col, negative_charges.height))
            df = df.filter(pl.col(col).gt(0))
    return df

def remove_equal_pickup_dropoff_times(df:pl.DataFrame, pu_col:str, do_col:str, logger_obj:logging.Logger) -> pl.DataFrame:
    equal_pu_do_dt = df.filter(pl.col(pu_col).ge(pl.col(do_col)))
    logger_obj.info("Taxi trips without duration: {0}".format(equal_pu_do_dt.height))
    df = df.filter(pl.col(pu_col).lt(pl.col(do_col)))
    return df

def feature_engineer_trip_duration(df:pl.DataFrame,  pu_col:str, do_col:str, duratation_col_name:str) -> pl.DataFrame:
    df = df.with_columns(
        ( ( ( ( pl.col(do_col) - pl.col(pu_col) )/60 )/1_000_000 ).round(2)).cast(pl.Float64).alias(duratation_col_name)
    )
    return df

def feature_engineer_trip_hour(df:pl.DataFrame, cols:list) -> pl.DataFrame:
    for col in cols:
        df = df.with_columns(
            pl.col(col[0]).dt.hour().cast(pl.Int64).alias("{0}_hour".format(col[1])),
        )
    return df

def feature_engineer_trip_daytime(df:pl.DataFrame, daytime_mapper:list, cols:tuple) -> pl.DataFrame:
    for col in cols:
        df = df.with_columns(
            pl.col(col[0]).map_elements(daytime_value, return_dtype=pl.Utf8).map_dict(daytime_mapper).cast(pl.Int64).alias("{0}_daytime".format(col[1]))
        )
    return df

def retrieve_latest_modified_file(relative_path:str, short_version:bool, samples_values:str=None):
    if short_version:
        json_files = glob.glob("{0}/*_data_{1}.json".format(relative_path, samples_values))
    else:
        json_files = glob.glob("{0}/*_data.json".format(relative_path))
    files_with_timestamps = []
    for file in json_files:
        modified_time = os.path.getmtime(file)
        files_with_timestamps.append((file, modified_time))
    latest_json_file = max(files_with_timestamps, key=lambda x: x[1])[0]
    return latest_json_file