from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage import filters

def plotDifference(original, new, title1='Original Image', title2='New Image'):
    """
    Function to plot and compare two images side by side.
    
    Parameters:
    original (ndarray): The original image.
    new (ndarray): The new or modified image.
    """

    plt.subplot(121),plt.imshow(original,cmap = 'gray')
    plt.title(title1), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(new,cmap = 'gray')
    plt.title(title2), plt.xticks([]), plt.yticks([])
    plt.show()

def detectionMorph(input, plot=False, print_results=True, polylines=None, img_name=None):
    """
    Function to detect stitches in an image using morphological operations and edge detection.

    Parameters:
    input (ndarray): Input image in which stitches are to be detected.
    plot (bool): Flag to plot intermediate results. Default is False.
    print_results (bool): Flag to print the number of detected stitches and compare with ground truth if provided. Default is True.
    polylines (dict): Dictionary containing ground truth stitch information, used if print_results is True.
    img_name (str): Image name key for accessing ground truth data in polylines, used if print_results is True.

    Returns:
    tuple: Number of detected stitches and the list of stitch contours.
    """

    img = input.copy()

    edges = cv2.Canny(img,100,200)

    #morphological operations - closing
    kernel = np.ones((2,4),np.uint8)
    closing = cv2.morphologyEx(edges.copy(), cv2.MORPH_CLOSE, kernel)

    #morphological operations - erosion
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(closing.copy(),kernel,iterations = 1)

    #delete noise
    kernel = np.ones((5,1),np.uint8)
    opening = cv2.morphologyEx(erosion.copy(), cv2.MORPH_OPEN, kernel)

    #connect lines that are close to each other
    kernel = np.ones((10,1),np.uint8)
    dilation = cv2.dilate(opening.copy(),kernel,iterations = 1)

    #delete objects that are in left 5% of the image
    dilation[:,0:int(dilation.shape[1]/20)] = 0

    #connect lines that are above each other
    kernel = np.ones((11,9),np.uint8)
    final = cv2.dilate(dilation.copy(),kernel,iterations = 1)

    # Nalezení kontur v obraze
    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrace kontur podle velikosti a tvaru, aby odpovídaly stehům
    stitches = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 180 < area < 700:  # Nastavení rozsahu velikosti kontury podle vašich potřeb
            stitches.append(contour)


    if plot:
        plotDifference(img, edges, 'Original Image', 'canny - Edges')
        plotDifference(edges, closing, 'canny - Edges', 'Closing')
        plotDifference(closing, erosion, 'Closing', 'Erosion')
        plotDifference(erosion, opening, 'Erosion', 'Opening')
        plotDifference(opening, dilation, 'Opening', 'Dilation')
        plotDifference(dilation, final, 'Dilation', 'Final')
        #plot stitches
        img = cv2.drawContours(img.copy(), stitches, -1, (255,0,0), 2)
        plotDifference(final, img, 'Final', 'Stitches - contours')


    if print_results:
        print(f'Number of stitches: {len(stitches)}')

        if polylines is not None and img_name is not None:
            print(f'Real number of stitches: {len(polylines[img_name]["stitches"])}')

    return len(stitches), stitches

def detectionRobertsSobel(img, threshold=0.2, plot=False, print_results=False, filter='roberts'):
    """
    Function to detect edges in an image using Roberts or Sobel filter, followed by morphological operations to refine detection.
    
    Parameters:
    img (ndarray): Input image in which edges are to be detected.
    plot (bool): Flag to plot intermediate results. Default is False.
    print_results (bool): Flag to print the number of detected contours (stitches). Default is False.
    filter (str): Edge detection filter to use ('roberts' or 'sobel'). Default is 'roberts'.
    
    Returns:
    tuple: Number of detected contours and the list of contours.
    """

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = gimg
    edge_roberts = filters.roberts(image)
    edge_sobel = filters.sobel(image)

    #choose filter
    edge = edge_roberts if filter == 'roberts' else edge_sobel

    #dilation
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(edge.copy(),kernel,iterations = 2)

    #erosion
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(dilation.copy(),kernel,iterations = 2)

    #use erosion to delete horizontal lines
    kernel = np.ones((5,1),np.uint8)
    test = cv2.erode(erosion.copy(),kernel,iterations = 2)

    #transform to 1 and 0
    test2 = test > threshold

    #find contours
    contours, hierarchy = cv2.findContours(test2.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #plot
    if plot:
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                            figsize=(8, 4))

        axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
        axes[0].set_title('Roberts Edge Detection')

        axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
        axes[1].set_title('Sobel Edge Detection')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        plotDifference(edge_roberts, dilation)
        plotDifference(dilation, erosion)
        plotDifference(erosion, test)
        plotDifference(test, test2)


    if print_results:
        #number of contours
        print(f'Number of stitches: {len(contours)}')

    return len(contours), contours
    

def blurredCanny(img, plot=False, print_results=False):
    """
    Function to detect stitches in an image using Gaussian blur and Canny edge detection,
    followed by morphological operations to refine detection.

    Parameters:
    img (ndarray): Input image in which stitches are to be detected.
    plot (bool): Flag to plot intermediate results. Default is False.
    print_results (bool): Flag to print the number of detected contours (stitches). Default is False.

    Returns:
    tuple: Number of detected contours and the list of contours.
    """

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # Předzpracování obrazu (např. aplikace Gaussova rozostření)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detekce hran pomocí Canny edge detectoru
    edges = cv2.Canny(blurred, 50, 150)

    #dilatace
    kernel = np.ones((4,1),np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    #contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #delete small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    #make img from contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)

    #erosion
    kernel = np.ones((15,1),np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)

    #dilation
    kernel = np.ones((15,4),np.uint8)
    dilated2 = cv2.dilate(eroded, kernel, iterations=1)

    #contours
    contours, _ = cv2.findContours(dilated2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #plot
    if plot:
        plotDifference(img, edges)
        plotDifference(edges, dilated)
        plotDifference(dilated, mask)
        plotDifference(mask, eroded)
        plotDifference(eroded, dilated2)

    #print results
    if print_results:
        print(f'Number of stitches detected: {len(contours)}')

    return len(contours), contours