#!/usr/bin/env python

import polars as pl
import requests
import boto3
import logging
import configparser
import os
import json
import sys
from io import StringIO
from datetime import datetime
sys.path.append('./src')
from custom_logger import setup_logger

def main(logger_object:logging.Logger):
    
    # Initialize configparser object
    config = configparser.ConfigParser()
    execution_timestamp = datetime.now().strftime('%Y%m%d')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))

    #========================================================
    # NYC ZONES COLLECTION FROM EXTERNAL API
    #========================================================
    api_url = config.get("api-settings", "nyc_zones_api")
    params = {
        "$limit": 1000,
        "$$app_token": config.get("api-settings", "app_token")
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if not data:
            sys.exit()
        with open("{0}/data/geospatial/nyc_zone_districts_{1}_data.json".format(parent_dir, execution_timestamp), "w") as f:
            json.dump(data, f, indent=4)
    else:
        logger_object.error("API request failed.")
        logger_object.error("Error: {0}".format(response.status_code))
        logger_object.error(response.text)
    
    logger_object.info("EXTRACTION FINISHED - Data collection of NYC Zone Districts from the Socrata API completed.")

if __name__ == "__main__":
    log_filename = f"nyc_zones/nyc_zones_collection_execution_logs_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    logger = setup_logger(log_filename)

    try:
        main(logger)
        logger.info("SUCCESS: NYC zones collection completed.")
    except Exception as e:
        logger.error(e)
        logger.error("FAIL: NYC zones collection failed.")