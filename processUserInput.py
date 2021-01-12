#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Nov  7 22:28:30 2020

@author: Mask Mandator
@author: Bryan Yakimsky
@author: Jacob Zietek
'''

from API_KEYS import main as API_LIST
from cloudant import cloudant
from cloudant import cloudant_iam
import numpy as np
from google.cloud import automl
from google.cloud import vision
from google.cloud import bigquery
from PIL import Image, ImageOps
import os, io, argparse

project_id, model_id, DB_USERNAME, DB_API_KEY = API_LIST()


# This string is used to query a student via ID number. Concatenate
# with an id number at the end and run a query.

QUERY_DEFAULT = """
    SELECT * FROM `maskmandator-294920.purdue.purdue` WHERE id = 
"""

# Example of how to update num infractions using SQL query.

'''
SET numInfractions = '1'
WHERE CustomerID = 1;
'''

def get_user_info_by_id(id):
    '''
    Retrieves row of data from a Google Cloud BigQuery DB for a student via an ID number.
    
    :params id: ID number to be queried.
    :type id: Integer
    :rtype: cloud.bigquery.job.query.QueryJob
    :return: Returns a row pertaining to a student from a BigQuery database.
    '''
    
    client = bigquery.Client()
    query_job = client.query(QUERY_DEFAULT + id)
    
    return query_job


def is_wearing_mask(file_path):
    '''
    Uses Trained Google Cloud AutoML Model to determine if an individual is wearing a face mask

    :params file_path: The path to the local image file.
    :type file_path: string
    :rtype: boolean
    :return: Returns a boolean that describes if a user is wearing a mask True if wearing mask and False if not wearing a mask, or inclunsive
    '''
    prediction_client = automl.PredictionServiceClient()

    # Get the full path of the model.
    model_full_id = automl.AutoMlClient.model_path(
        project_id, "us-central1", model_id
    )

    # Read the file.
    with open(file_path, "rb") as content_file:
        content = content_file.read()

    image = automl.Image(image_bytes=content)
    payload = automl.ExamplePayload(image=image)

    # Params is additional domain-specific parameters.
    # score_threshold is used to filter the result
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#predictrequest
    params = {"score_threshold": "0.1"}

    request = automl.PredictRequest(
        name=model_full_id,
        payload=payload,
        params=params
    )
    response = prediction_client.predict(request=request)

    # Iterates through results of prediction
    for result in response.payload:
        # Checks for with_mask condition, and evalulates to true if there is a 99.99% certainty
        return True if (result.display_name == "with_mask" and result.classification.score > 0.9999) else False
    
    return False;


def get_crop_hint(path):
    '''Localize objects in the local image.

    :param path: The path to the local image file.
    :type image_file: string
    :rtype: 2D List
    :return: Returns the vertices for the bounding of the hint (Face)
    '''

    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    for face in faces:

        vertices = ([[vertex.x, vertex.y]
            for vertex in face.bounding_poly.vertices])

        return vertices

    return None;

def crop_to_hint(image_file):
    '''Crops the image using the hints in the vector list.

    :param image_file: The path to the local image file.
    :type image_file: string
    :rtype: void

    Saves the cropped image as a file
    '''

    vects = get_crop_hint(image_file)

    if(vects is None):
        im = Image.open(image_file)
        im.save('output-crop.png','PNG')
        return 


    im = Image.open(image_file)
    width, height = im.size
    im2 = im.crop([vects[0][0] -10, vects[0][1] - 10,
                  vects[2][0] + 10, vects[2][1] + 10])

    im2.save('output-crop.png', 'PNG')

def main():
    '''
    Takes a picture and determines if a person present in the image is wearing a mask

    :rtype: string
    :return: An output string that describes whether the user is wearing a mask
    '''

    # Gets and sets API Keys
    project_id, model_id, DB_USERNAME, DB_API_KEY = API_LIST()
    # Sets environment from Google Cloud Service Token
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'MaskMandator-a0512b925076.json'
    
    #File Path for Image to be Processed
    file_path = "face.png"
    
    #Boolean that describes if the subject is wearing a mask (True for wearing mask or False for not wearing mask)
    wearingMark = is_wearing_mask(file_path)

    if(not wearingMark):
        return("Please put on your mask! You have been penalized.")
    
    return("Access granted. Thank you, and stay safe!")


        
if __name__ == '__main__':
    main()
    
