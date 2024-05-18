import random

def randomStitches(img_names, images, polylines, print_results=False):
    """
    Function to randomly determine the number of stitches in images and compare 
    with the actual number of stitches, then calculate the accuracy.
    
    Parameters:
    img_names (list): List of image names.
    images (dict): Dictionary of images where key is image name and value is the image data.
    polylines (dict): Dictionary of polylines where key is image name and value contains stitch information.
    print_results (bool): Flag to print the results of the accuracy calculation. Default is False.
    
    Returns:
    int: The number of correct detections.
    """

    correct = 0

    # Loop through each image
    for img_name in img_names:
        img = images[img_name].copy()
        edges_number = random.randint(1, 6)

        # Check if the number of stitches is equal to the randomly generated number
        if edges_number == len(polylines[img_name]["stitches"]):
            correct += 1

    # Print the results
    if print_results:
        print(f'Correct detections: {correct}/{len(img_names)}')
        print(f'Accuracy: {correct/len(img_names)*100}%')

    return correct