#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Nov  7 22:28:30 2020

@author: Mask Mandator
@author: Bryan Yakimsky
@author: Jacob Zietek
'''

from API_KEYS import main as API_LIST
import numpy as np
from google.cloud import automl
from google.cloud import vision
from google.cloud import bigquery
from PIL import Image, ImageOps
import os, io, argparse
from picamera import PiCamera
from time import sleep


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
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken_GoogleVision.json'
    client = bigquery.Client()
    query_job = client.query(QUERY_DEFAULT + id)
    
    return query_job["name"], query_job["hasVaccine"], query["numInfractions"]


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
        im.save(image_file,'PNG')
        fourier_transform(image_file)
        return 


    im = Image.open(image_file)
    width, height = im.size
    im2 = im.crop([vects[0][0] -10, vects[0][1] - 10,
                  vects[2][0] + 10, vects[2][1] + 10])

    im2.save(image_file, 'PNG')
    fourier_transform(image_file)
    



def fourier_transform(file_path):
    '''Converts an image to a the magnitude spectrum by using fourier transformations, applys a disk filter, and converts back to image

    :params file_path: The path to the local image file.
    :type file_path: string
    :rtype: void

    Saves the cropped image as a file
    '''
    import numpy as np
    import cv2 
    from matplotlib import pyplot as plt
    from PIL import Image


    img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # center

    # Concentric BPF mask,with are between the two cerciles as one's, rest all zero's.
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    r_out = 80
    r_in = 5
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]

    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                            ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1

    # apply mask@ext:ms-vscode-remote.remote-ssh,ms-vscode-remote.remote-ssh-edit config file and inverse DFT
    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])



    #cv2.imwrite(r"gray.bmp",img2)




    plt.imshow(img_back, cmap='gray')
    plt.axis('off')


    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0)

def take_picture():
    camera = PiCamera()

    sleep(2) #Allows brightness adjustment, makes camera quality better
    camera.capture('/home/pi/Documents/main/face.png')

def main():
    '''
    Takes a picture and determines if a person present in the image is wearing a mask

    :rtype: string
    :return: An output string that describes whether the user is wearing a mask
    '''
    
    take_picture()
    # Gets and sets API Keys
    project_id, model_id, DB_USERNAME, DB_API_KEY = API_LIST()
    # Sets environment from Google Cloud Service Token
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken_GoogleVision.json'
    
    #File Path for Image to be Processed
    file_path = "face.png"
    crop_to_hint(file_path)

    
    #Boolean that describes if the subject is wearing a mask (True for wearing mask or False for not wearing mask)
    wearingMark = is_wearing_mask(file_path)

    name, vaccineStatus, numInfractions = get_user_info_by_id("1")
    
    if(not wearingMark):
        print("Please put on your mask {}! Try again.", name)
    else:
        if(numInfractions > 2): # More than 3 infractions leads to disciplinary action
            print("You have {} infractions on your account {}, please clear these up in the main office before entering.", numInfractions, name)
        else:
            if(vaccineStatus == "False"):
                print("Access granted. Thank you {}, and stay safe! Please get COVID-19 vaccine when available.", name)
            else:
                print("Access granted. Thank you {}, and stay safe!", name)

        
if __name__ == '__main__':
    main()
    
