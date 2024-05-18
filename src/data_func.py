import xmltodict
import os
import os
from skimage import io
from skimage.color import rgba2rgb, gray2rgb
import cv2
import numpy as np

def read_file(file):
    """
    Reads an XML file and converts its contents into a dictionary.

    Parameters:
    file (str): The path to the XML file to be read.

    Returns:
    dict: A dictionary representation of the XML file contents.
    """

    with open('annotations.xml', 'r', encoding='utf-8') as file: 
        xml = file.read() 
    
    annotations = xmltodict.parse(xml) 

    print('Successfully read the file.')

    return annotations

def loadImages(path):
    """
    Loads images from a specified folder into a dictionary.

    Parameters:
    path (str): The path to the folder containing the images.

    Returns:
    dict: A dictionary where keys are image filenames and values are image arrays.
    """

    images = {}

    # Loop through the files in the folder and read the images
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):  # Ensure it's a file
            img = io.imread(file_path)

            if img.ndim == 2:  # Convert grayscale to RGB
                img = gray2rgb(img)
                
            elif img.shape[-1] == 4:  # Check if image has an alpha channel
                img = rgba2rgb(img)
            images[filename] = img

    print('Successfully loaded the images.')

    return images


def extractPolyline(annotations):
    """
    Extracts polyline data from annotations and organizes it into a dictionary.

    Parameters:
    annotations (dict): A dictionary containing annotation data.

    Returns:
    dict: A dictionary where keys are image names and values are lists of points representing polylines.
    """

    polyline = {}

    # Loop through the annotations and extract polyline data
    for obj in annotations['annotations']['image']:
        name = obj['@name'].strip('incision_couples/')
        stitches = []
        incision = []

        if 'polyline' not in obj:
            polyline[name] = {'incision': [], 'stitches': []}
            continue

        # Loop through the polyline objects
        for polyline_obj in obj['polyline']:
            
            # Check if the polyline object is a dictionary
            if not isinstance(polyline_obj, dict):
                continue
            
            # Extract the points from the polyline object
            if polyline_obj['@label'] == 'Incision':
                incision = polyline_obj['@points'].split(';')
                incision = [[int(float(point.split(',')[0])), int(float(point.split(',')[1]))] for point in incision]

            # Extract the stitch points from the polyline object
            elif polyline_obj['@label'] == 'Stitch':
                stitch = polyline_obj['@points'].split(';')
                #stitch = [point.split(',') for point in stitch]
                stitch = [[int(float(point.split(',')[0])), int(float(point.split(',')[1]))] for point in stitch]
                stitches.append(stitch)

        # Add the polyline data to the dictionary
        polyline[name] = {'incision': incision, 'stitches': stitches}

    print('Successfully extracted polyline data.')

    return polyline

def print_points(im_name, images, polyline, only_stitches = False):
    """
    Draws polylines on the given image based on incision and stitch points.

    Parameters:
    im_name (str): The name/key of the image in the images dictionary.
    images (dict): A dictionary containing image data, where keys are image names and values are image arrays.
    only_stitches (bool, optional): If True, only stitch points are drawn. If False, both incision and stitch points are drawn. Default is False.

    Returns:
    np.ndarray: The image array with the polylines drawn on it.
    """

    newIm = images[im_name].copy()
    incision_points = polyline[im_name]['incision']

    # Draw the incision points
    if not only_stitches:
        points = np.array(incision_points)
        pts = points.reshape(-1,1,2) # now shape [3,1,2]

        newIm = cv2.polylines(newIm, [pts], isClosed=False, color=(255,0,0), thickness = 1)

    # Draw the stitch points
    for stitch in polyline[im_name]['stitches']:
        points = np.array(stitch)
        pts = points.reshape(-1,1,2)
        newIm = cv2.polylines(newIm, [pts], isClosed=False, color=(0,255,0), thickness = 1)

    print('Successfully printed the points.')

    return newIm