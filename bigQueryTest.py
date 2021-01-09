#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:41:55 2021

@author: jacobzietek
"""

import os

from google.cloud import bigquery

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'MaskMandator-a0512b925076.json'

# Construct a BigQuery client object.
client = bigquery.Client()


query = """
    SELECT * FROM `maskmandator-294920.purdue.purdue` WHERE id = 1
"""

query_job = client.query(query)  # Make an API request.

print("The query data:")
for row in query_job:
    # Row values can be accessed by field name or index.
    print("id={}, name={}".format(row["id"], row["name"]))