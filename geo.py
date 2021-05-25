import gmsh # pip install --upgrade gmsh
# If gmsh is already installed (manual install using apt)
# the system will not recognize the python API. 
import numpy as np
from pandas import DataFrame

def init_all_points(imgDim, ContourDF, HierarchyDF, lc):
    '''
    Creates all domain points regarding ContourDF order.
    Returns a list of lists (all_points)
    Each item in all_points carries the tags associated 
    with each contour point, ordered according to dataframe info.
    '''
    # Store tags for later 
    all_points = []
    boundary_points = []

    # Define bounding box
    xmin = 0
    xmax = imgDim[0]
    ymin = 0
    ymax = imgDim[1]

    # Create points for bounding box
    Points =[]
    Points.append(gmsh.model.geo.addPoint(xmin, ymin, 0, lc))
    boundary_points.append([xmin, ymin, 0, 'left-bottom', Points[-1], 'BoundingBox'])
    Points.append(gmsh.model.geo.addPoint(xmax, ymin, 0, lc))
    boundary_points.append([xmax, ymin, 0, 'right-bottom', Points[-1], 'BoundingBox'])
    Points.append(gmsh.model.geo.addPoint(xmax, ymax, 0, lc))
    boundary_points.append([xmax, ymax, 0, 'right-top', Points[-1], 'BoundingBox'])
    Points.append(gmsh.model.geo.addPoint(xmin, ymax, 0, lc))
    boundary_points.append([xmin, ymax, 0, 'left-top', Points[-1], 'BoundingBox'])

    # Sweep all contours
    for index, row in ContourDF.iterrows():
        currentContour = ContourDF.values[index][0]
        currentHierarchy = HierarchyDF.values[index]

        Points = []

        # If currentContour has parent flip its direction
        if currentHierarchy[3] > 0:
            currentContour = currentContour[::-1]

        for point in currentContour:
            
            x = point[0]
            y = point[1]
            z = point[2]

            # Check if contour touches the boundaries
            bottom = np.isclose(y,ymin, rtol = 0.01, atol = 0.01)
            right = np.isclose(x,xmax, rtol = 0.01, atol = 0.01)
            top = np.isclose(y,ymax, rtol = 0.01, atol = 0.01)
            left = np.isclose(x,xmin, rtol = 0.01, atol = 0.01)

            if bottom or right or top or left:
                if bottom:
                    y = ymin
                    # Create point in gmsh
                    Points.append(gmsh.model.geo.addPoint(x, y, z, lc))
                    boundary_points.append([x, y, z, 'bottom', Points[-1], index])
            
                if right:
                    x = xmax
                    # Create point in gmsh
                    Points.append(gmsh.model.geo.addPoint(x, y, z, lc))
                    boundary_points.append([x, y, z, 'right', Points[-1], index])
            
                if top:
                    y = ymax
                    # Create point in gmsh
                    Points.append(gmsh.model.geo.addPoint(x, y, z, lc))
                    boundary_points.append([x, y, z, 'top', Points[-1], index])
            
                if left:
                    x = xmin
                    # Create point in gmsh
                    Points.append(gmsh.model.geo.addPoint(x, y, z, lc))
                    boundary_points.append([x, y, z, 'left', Points[-1], index])
            
            else:
                Points.append(gmsh.model.geo.addPoint(x, y, z, lc))
            
        all_points.append(Points)
      
    # Convert boundary points into dataframe
    boundary_data = DataFrame(boundary_points, columns = ['x','y','z', 'boundary','Index','ContourTag'])
    
    return all_points, boundary_data


def init_domain_boundary_contours(boundary_data, sweep_string = ['left-bottom','bottom','right-bottom','right','right-top','top','left-top','left']):
    '''
    Auxiliary function to create rectangle domain according to points on boundaries. 
    Returns a multi-object list with the boundary line tag and its respective building points.

    boundary_data -> pandas dataframe with x, y and z coordinates, boundary and contourTag
    
    Default direction: ['left-bottom','bottom','right-bottom','right','right-top','top','left-top','left']
    '''
    
    boundary_points_ordered = []

    for direction in sweep_string:
        content = boundary_data[boundary_data['boundary']== direction].copy()
        
        # Sort according to boundary
        # if direction == 'bottom' or direction == 'right' or direction == 'top' or direction == 'left':
        if direction == 'bottom':
            content.sort_values(by = ['x'], inplace = True)

        if direction == 'right':
            content.sort_values(by = ['y'], inplace = True)

        if direction == 'top':
            content.sort_values(by = ['x'], inplace = True, ascending = False)

        if direction == 'left':
            content.sort_values(by = ['y'], inplace = True, ascending = False)

        for item in content.values:
            boundary_points_ordered.append(item[4])
    
    boundary_points_ordered.append(boundary_points_ordered[0])
    lines = []

    # Create boundary lines and store its respective information
    for i in range(0,len(boundary_points_ordered)-1):
        p0 = boundary_points_ordered[i]
        p1 = boundary_points_ordered[i+1]
        lines.append([gmsh.model.geo.addLine(p0,p1), (p0,p1)])

    return lines

def init_all_lines(all_points, boundary_data):
    '''
    Create lines in domain. Sweeps each item in all_points list and creates 
    a line from n_th to (n+1)_th points. An auxiliary function is called to create domain boundary.
    '''
    # Create lines 
    all_lines = []

    # ext_lines_control is a multi-object list -> [lineTag, (point1Tag, point2Tag)]
    ext_lines_control = init_domain_boundary_contours(boundary_data) # Auxiliary function call
    ext_lines = []
    bpoints = []
    
    # Separate items to ease the check
    for item in ext_lines_control:
        ext_lines.append(item[0])
        bpoints.append(item[1])
        
    all_lines.append(ext_lines)

    for contour_points in all_points:
        lines = []

        # Append first point in last position of contour_points
        contour_points.append(contour_points[0])
        #? Revert order to create line loop properly (no idea why - think about another way of doing it)
        contour_points = contour_points[::-1]

        for i in range(0,len(contour_points)-1):

            p1 = contour_points[i+1]
            p0 = contour_points[i]

            # Check if line is already created
            if (p0,p1) in bpoints or (p1,p0) in bpoints:
                # Get (p0,p1) position
                if (p0,p1) in bpoints:
                    index = bpoints.index((p0,p1))
                if (p1,p0) in bpoints:
                    index = bpoints.index((p1,p0))
                # Append already created line
                lines.append(ext_lines[index])
            else:
                lines.append(gmsh.model.geo.addLine(p0,p1))

        # store the tags according to Contour order in dataframe
        all_lines.append(lines)
    
    return all_lines

def init_all_line_loops(all_lines):

    all_line_loops = []

    for contour_lines in all_lines:
        ll = [line for line in contour_lines]
        all_line_loops.append(gmsh.model.geo.addCurveLoop(ll))
    
    return all_line_loops
        

def init_all_surfaces(all_line_loops, HierarchyDF):
    '''
    Creates plane surfaces for the mapped regions according to predefined line loops.
    Surfaces within surfaces are created according to hierarchy data.
    '''

    all_surfaces = []

    # Create exterior surface
    ext_ll = []
    ext_ll.append(all_line_loops[0])

    # Deleting first element from all_line_loops
    del(all_line_loops[0])

    # Filter data
    outermost_clls_tags = HierarchyDF[HierarchyDF['Parent'] < 0].index

    for tag in outermost_clls_tags:
        outermost_ll = all_line_loops[tag]
        ext_ll.append(outermost_ll)
    
    all_surfaces.append(gmsh.model.geo.addPlaneSurface(ext_ll))

    # Create contour surfaces:
    for index, ll in enumerate(all_line_loops):

        # Check presence of internal contour
        hierarchy = HierarchyDF.values[index]
        lineloop = []
        lineloop.append(ll)

        if hierarchy[2] > 0:
            # Surface is defined as the region inside parent contour and 
            # Outside children contours 
            
            # Get children contour tags
            children = HierarchyDF[HierarchyDF['Parent'] == index].index

            for item in children:
                loop = all_line_loops[item]
                lineloop.append(loop)
        
        all_surfaces.append(gmsh.model.geo.addPlaneSurface(lineloop))
        
    return all_surfaces

def create_physical_surfaces(all_surfaces, Colors, Clusters):
    '''
    Creates physical plane surface according to image segmentation. 
    '''
    physDim = 2
    # Create physical regions dataframe
    data = {'Surfaces': all_surfaces, 'ColorTag': Colors}
    phys_df = DataFrame(data)

    # Create physical regions according to number of clusters
    for i in range(0, Clusters):
        # i is the color tag
        reg = phys_df[phys_df['ColorTag'] == i].values
        reg = [item[0] for item in reg] # Get all surfaces with respective color tag

        # Create physical region
        ps = gmsh.model.addPhysicalGroup(physDim, reg)
        model_name = 'region_'+str(i)
        gmsh.model.setPhysicalName(physDim, ps, model_name)

def create_physical_lines(all_lines, boundary_data, boundary_tags):
    '''
    Create physical lines for the boundaries, named according to boundary_tags.
    '''
    # Get boundary lines
    blines = all_lines[0]

    pass


def create_mesh(Cdata, Hdata, Colors, imgName, Clusters, imgDim):
    '''
    Initialize gmsh and creates all entities according to image data.

    '''
    # Initialize  gmsh
    gmsh.initialize()

    # Output messages on the terminal
    gmsh.option.setNumber("General.Terminal", 1)

    # Add model name
    gmsh.model.add(imgName)

    lc = 10 # Characteristic length of elements 

    # Creating geometry 
    all_points, boundary_data = init_all_points(imgDim, Cdata, Hdata, lc)
    all_lines = init_all_lines(all_points, boundary_data)
    all_line_loops = init_all_line_loops(all_lines)
    all_surfaces = init_all_surfaces(all_line_loops, Hdata)

    # Create physical groups: Lines (rectangular domain)
    boundary_tags = ['left', 'bottom', 'right', 'top']
    create_physical_lines(all_lines, boundary_data, boundary_tags) #! IMPLEMENT   

    # Create physical groups: Surfaces
    create_physical_surfaces(all_surfaces, Colors, Clusters)

    # Synchronize gmsh data structure: calling API for the built-in kernel
    gmsh.model.geo.synchronize()

    # Generate .msh file
    gmsh.model.mesh.generate(2)

    # Save mesh
    mesh = imgName+'.msh' # Remove .png extension and add .msh
    gmsh.write(mesh)
    gmsh.fltk.run()
    gmsh.finalize()