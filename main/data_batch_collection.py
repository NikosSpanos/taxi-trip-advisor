#!/usr/bin/env python

import requests
import boto3
import sys
import logging
import configparser
import os
import json
from io import StringIO
from datetime import datetime, time, timedelta
sys.path.append('./src')
from custom_logger import setup_logger
from landing_modules import date_calculation
    
def main(logger_object:logging.Logger):
    
    if len(sys.argv) !=3:
        logger_object.error("Usage: python data_batch_collection.py date_value date_interval")
        sys.exit(1)
    
    # Initialize module arguments
    initial_date = sys.argv[1]
    date_interval = int(sys.argv[2])
    current_year = datetime.now().year
    time_object = time(0, 0)

    # Initialize configparser object
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))

    #========================================================
    # SETUP DATE INTERVAL FOR DATA EXTRACTION
    #========================================================
    try:
        input_year = int(initial_date.split("-")[0])
        input_month = int(initial_date.split("-")[1])
        input_day = int(initial_date.split("-")[2])

        if (input_year > current_year):
            logger_object.error("Invalid year value. Year cannot be greater than the current fiscal year.")
            sys.exit(1)
        if (input_month > 12):
            logger_object.error("Invalid month value. Month should be a positive integer between 1-12.")
            sys.exit(1)
        if (input_day < 0 or input_day > 31):
            logger_object.error("Invalid day value. Day should be a positive integer between 1-31.")
            sys.exit(1)
        starting_date_object = datetime.strptime(initial_date, "%Y-%m-%d")
        starting_date = datetime.combine(starting_date_object.date(), time_object)
        ending_date = date_calculation(starting_date, date_interval)

        #Apply the floating_timestamp format for querying the taxi trip data
        starting_date = starting_date.strftime("%Y-%m-%dT%H:%M:%S")
        ending_date = ending_date.strftime("%Y-%m-%dT%H:%M:%S")
        logger_object.info("Extracting data from: {0} to: {1}".format(ending_date, starting_date))

    except ValueError:
        logger_object.error("Invalid arguments. Please use YYYY-MM-DD format and for intervals use positive integer values.")

    #========================================================
    # DATA COLLECTION FROM EXTERNAL API
    #========================================================
    api_url = config.get("api-settings", "collection_api")
    batch_size = int(config.get("api-settings", "batch_size"))
    total_records = int(config.get("api-settings", "total_records"))
    execution_timestamp = datetime.now().strftime('%Y%m%d')
    total_iterations = total_records / batch_size
    iteration_value = 0
    offset = 0
    logger_object.info("EXTRACTION STARTED - Data collection from the Socrata API started.")

    while offset < total_records:
        params = {
            "$limit": batch_size,
            "$offset": offset,
            "$$app_token": config.get("api-settings", "app_token"),
            "$where": f"tpep_pickup_datetime <= '{starting_date}' and tpep_pickup_datetime > '{ending_date}'"
        }
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            with open("{0}/data/landing/yellow_taxi_trip_data_{1}_offset_{2}.json".format(parent_dir, execution_timestamp, offset), "w") as f:
                json.dump(data, f, indent=4)
            offset += batch_size
            logger_object.info("{0}/{1} offsets".format(offset, total_records))
            logger_object.info("{0}/{1} iterations ".format(iteration_value, total_iterations))
            iteration_value +=1
        else:
            logger_object.error("API request failed.")
            logger_object.error("Error: {0}".format(response.status_code))
            logger_object.error(response.text)
            break
    logger_object.info("EXTRACTION FINISHED - Data collection from the Socrata API completed.")

    #========================================================
    # DATA CONVERSION FROM JSON TO COLUMNAR TABLE
    #========================================================
    # schema = pl.DataFrame(all_records).schema
    # df = pl.DataFrame(all_records, schema=schema)
    # df.columns = list(map(lambda x: x.lower(), df.columns))
    # logger_object.info("Total records retrieved: {0}".format(df.height))

    #=============================================================
    # CREATE THE PARTITION COLUMN(s) to split the raw .json files
    #=============================================================
    # df = df.with_columns(pl.col("tpep_pickup_datetime").str.to_datetime("%Y-%m-%dT%H:%M:%S.000").dt.date().alias("partition_dt"))
    # partition_values = df.select(pl.col("partition_dt")).unique().to_dict(as_series=False)
    # logger.info("Unique partitions :{0}".format(partition_values["partition_dt"]))

    #========================================================
    # SAVE RAW DATA TO LADNING ZONE
    #========================================================
    # Establish connection to S3 bucket resource
    # s3_bucket_resource = boto3.resource(
    #     service_name  = "s3",
    #     region_name = config.get("settings", "aws_bucket_region"),
    #     aws_access_key_id = config.get("settings", "aws_access_key"),
    #     aws_secret_access_key = config.get("settings", "aws_secret_key"),
    # )
    #------------------------------------------------------------------------------------
    # bucket_name = config.get("settings", "aws_bucket_name")
    # bucket = s3_bucket_resource.Bucket(bucket_name)
    # execution_timestamp = datetime.now().strftime('%Y_%m_%d_00_00_00')
    # destination = config.get("settings", "aws_landing_unprocessed_folder") + "yellow_taxi_trip_sample_" + str(execution_timestamp) + ".json"
    # json_buffer = StringIO()
    
    #===============================================================
    # BENCHMARK 1 (execution time: 33:36)
    # Command: python src/data_batch_collection.py 2017-07-26 10
    #===============================================================
    # df.write_json(json_buffer)
    # try:
    #     # Retrieve the existing files in the bucket data/landing/unprocessed folder/
    #     for object_summary in bucket.objects.filter(Prefix=config.get("settings", "aws_landing_unprocessed_folder")):
    #         logger_object.info(object_summary.key)
    #     logger_object.info("WRITE STARTED - Writing JSON file with collected under path : {0}".format(destination))
    #     bucket.put_object(Bucket = bucket_name, Key = destination, Body = json_buffer.getvalue())
    #     logger_object.info("WRITE COMPLETED - Data have been written to landing zone under path: {0}".format(destination))
    #     for object_summary in bucket.objects.filter(Prefix=config.get("settings", "aws_landing_unprocessed_folder")):
    #         logger_object.info(object_summary.key)
    # except Exception as e:
    #     logger_object.error(e)
    #     logger_object.error("WRITE TO LANDING ZONE FAILED - Check execution logs for errors.")

if __name__ == "__main__":
    log_filename = f"batch_collection/batch_collection_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(log_filename)

    try:
        main(logger)
        logger.info("SUCCESS: Batch collection process completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: Batch collection process failed.")
