from geo import create_mesh
from imgSeg import get_contours_in_df
import os

##### Load image
imgName = 'TestData4.png'   
# Number of colors (segments) inside domain
Clusters = 2
# Segment image and create topology dataframe
Cdata, Hdata, Colors, imgDim = get_contours_in_df(imgName, Clusters)

# Removing filename extension
imgName, imgExtension = os.path.splitext(imgName)

# Create mesh file
create_mesh(Cdata, Hdata, Colors, imgName, Clusters, imgDim)

#TODO: Create auxiliar method to define external physical lines
#TODO: Rescale mesh according to user
#TODO: Refine mesh according to user 
