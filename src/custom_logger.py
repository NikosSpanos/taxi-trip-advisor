#!/usr/bin/env python

import logging
import os

def setup_logger(log_filename):
    logging.basicConfig(
        filename=os.path.join("logs", log_filename),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)