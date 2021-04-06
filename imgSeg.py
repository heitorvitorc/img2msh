import cv2
import pandas as pd
import numpy as np
import os

def segment_image(imageObj, k):
    '''
    Sharp image interfaces. 
    '''
    
    # # Get Image Filename
    # image_file = imageObj.filename
    
    # Read Image
    #? Mudar para versão do App 
    image = cv2.imread(os.path.join(imageObj))
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    
    # convert to float
    pixel_values = np.float32(pixel_values)
    
    # Define Stopping Criteria (?)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Number of pixels of certain colors
    colors, counts = np.unique(image.reshape(-1, 3),
                                return_counts = True, 
                                axis = 0)

    # number of clusters (K)
    if k <= len(colors):
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids and back to original shape
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def formatContours(ContourDataFrame):
    """
    Format contours into 3D coordinate system. 
    Returns a contour dataframe.
    """
    Contours = []

    for index, row in ContourDataFrame.iterrows():
        
        currentContour = ContourDataFrame.values[index][0]
        Rcont = []

        # for j in range(0,len(currentContour)):
        for point in currentContour:
            Rcont.append([point[0][0], point[0][1], 0])

            # aux = currentContour[j]
            # Rcont.append([  aux[0][0], aux[0][1],   0.0])
                
        Contours.append(np.array(Rcont))

    # Rewrite formatted dataframe
    Cdata = pd.DataFrame(Contours)

    return Cdata


def get_contour_orientation(contours):
    '''
    Get contour orientation.
    '''

    for contour in contours:
        # get contour centroid coordinates
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])  

        for i, point in enumerate(contour):
            # Create vectors:
            v_centroid = cx
            #! CONTINUAR FUNÇÃO


def create_random_bounded(x, y, w, h, samples):
    '''
    Creates discrete normal distribution of points bounded inside
    (x, x+w) and (y, y+h). Returns a tuple of random coordinates.

    '''
    # samples = 100
    randx = np.random.randint(low =x, high = x+w, size = samples)
    randy = np.random.randint(low =y, high = y+h, size = samples)
    # Convert to tuple
    rand = tuple(map(tuple, np.vstack((randx, randy)).T))

    return rand


def get_contour_tag(Cdata, Hdata, gs):
    '''
    This function returns the color tag from inside contour area.
    If outermost contour contains children contours, returns the color tag
    from the region between outermost and children contours.

    Input:
    Cdata -> Contour coordinate system dataframe
    Hdata -> Hierarchy dataframe
    gs -> Grayscaled image

    Output:
    Colors -> Color tag ordered by contour index in Cdata
    '''

    Tag = []

    # Get Color tag from outermost region (outside contours)
    x = 0; y = 0; w = gs.shape[1]; h = gs.shape[0]
    samples = 300
    rand = create_random_bounded(x, y, w, h, samples)

    # Get Outermost contours
    outermost_contours = Hdata[Hdata['Parent'] < 0].index

    for pixel in rand:
        # Check if pixel is inside any of the outermost contours. cv2.pointPolygonTest returs positive value if point
        # is inside contour, zero if point is in contour and negative value if point is outside contour area.
        inside_pol = [cv2.pointPolygonTest(Cdata.values[out][0], pixel, True) for out in outermost_contours]

        if all(item < 0 for item in inside_pol):
            Tag.append(pixel)
            break
    
    # Get Color tag from region in / in between contours:
    for index, row in Cdata.iterrows():
        
        currentContour = Cdata.values[index][0]
        currentHierarchy = Hdata.values[index]

        # Create contour bounding box
        x, y, w, h = cv2.boundingRect(currentContour)

        # Create random tuples from a discrete uniform distribution
        samples = 100
        rand = create_random_bounded(x, y, w, h, samples)

        # Sweep random values inside bounding box
        for pixel in rand:
        
            # Check if pixel is inside currentContour
            insideParent = cv2.pointPolygonTest(currentContour, pixel, True) # Returns positive if pixel is inside contour, negative if not, zero if on contour

            if insideParent < 0:
                continue

            # Check which contour has currentContour as parent
            if currentHierarchy[2] > 0:

                # Get childreContours indexes
                children = Hdata[Hdata['Parent'] == index].index # returns a pandas iterable structure

                # Check if pixel is inside childrenContours
                insideChildren = [cv2.pointPolygonTest(Cdata.values[chIndex][0], pixel, True) for chIndex in children]

                # Check if pixel is inside currentContour and outside all childrenContours
                if insideParent > 0 and all(ch < 0 for ch in insideChildren):
                    
                    Tag.append(pixel)
                    break
            
            elif insideParent > 0:
                
                Tag.append(pixel)
                break
    
    
    # Convert Tag into color segment: Tag[(x,y)] -> color[(y,x)]
    Colors =[]

    for i in range(0,len(Tag)):
        x = Tag[i][0]
        y = Tag[i][1]
        Colors.append(gs[y][x])
    
    return Colors

def get_contours_in_df(imgName, Clusters):
    '''
    Call auxiliary functions to process contour info into a dataframe.
    
    '''

    img = segment_image(imgName,Clusters)
    # Convert image in grayscale
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##### Get contours
    contours, hierarchy = cv2.findContours(gs,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##### Create pandas dataframe for contour and hierarchy
    Cdata = pd.DataFrame(contours,columns =['Contours'],dtype=object)
    Hdata = pd.DataFrame(hierarchy[0], columns = ['Next', 'Previous', 'First Child', 'Parent'], dtype=object)

    ##### Get Color info info
    Colors = get_contour_tag(Cdata, Hdata, gs)

    # Normalize Color tag
    Colors = [int(i/max(Colors)) for i in Colors]

    # Format the data
    Cdata = formatContours(Cdata)
    # Get domain dimension
    imgDim = (gs.shape[1], gs.shape[0])

    return Cdata, Hdata, Colors, imgDim