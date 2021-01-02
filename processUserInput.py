#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Nov  7 22:28:30 2020

@author: MaskMandator
'''

from API_KEYS import main as API_LIST
from cloudant import cloudant
from cloudant import cloudant_iam
import numpy as np
from google.cloud import automl
from google.cloud import vision
from PIL import Image, ImageOps
import os, io, argparse

project_id, model_id, DB_USERNAME, DB_API_KEY = API_LIST()



def isWearingMask(file_path):
    '''Uses Trained Google Cloud AutoML Model to determine if an individual is wearing a face mask

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

    # params is additional domain-specific parameters.
    # score_threshold is used to filter the result
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#predictrequest
    params = {"score_threshold": "0.1"}

    request = automl.PredictRequest(
        name=model_full_id,
        payload=payload,
        params=params
    )
    response = prediction_client.predict(request=request)

    #Iterates through results of prediction
    for result in response.payload:
        #Chcks for with_mask condition, and evalulates to true if there is a 99.99% certainty
        return True if (result.display_name == "with_mask" and result.classification.score > 0.9999) else False
    
    return False;

def get_crop_hint(path):
    '''Localize objects in the local image.

    :param path: The path to the local image file.
    :type image_file: string
    :rtype: NormalizedVertex
    :return: Returns the normalized vertices [0,1] for the bounding of the hint (QRCode In This Instance)
    '''

    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

    for object_ in objects:
        print(object_.name)
        if(object_.name == "2D barcode"):
            return object_.bounding_poly.normalized_vertices
    return None;

def crop_to_hint(image_file):
    '''Crops the image using the hints in the vector list.

    :param image_file: The path to the local image file.
    :type image_file: string
    :rtype: void

    Saves the cropped image as a file
    '''

    vects = get_crop_hint(image_file)


    im = Image.open(image_file)
    width, height = im.size
    im2 = im.crop([vects[0].x * width - 10, vects[0].y * height - 10,
                  vects[2].x * width + 10, vects[2].y * height + 10])
                  
    im2.save('output-crop.png', 'PNG')



def main():
    '''Takes a picture and determines if a person present in the image is wearing a mask

    :rtype: string
    :return: An output string that describes whether the user is wearing a mask
    '''

    #Gets and sets API Keys
    project_id, model_id, DB_USERNAME, DB_API_KEY = API_LIST()
    #Sets Environment from Google Cloud Service Token
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken_GoogleVision.json'
    
    #File Path for Image to be Processed
    file_path = "face.png"
    
    #Boolean that describes if the subject is wearing a mask (True for wearing mask or False for not wearing mask)
    wearingMark = isWearingMask(file_path)

    if(not wearingMark):
        return("Please put on your mask! You have been penalized.")
    return("Access granted. Thank you, and stay safe!")


        
if __name__ == '__main__':
    main()
    