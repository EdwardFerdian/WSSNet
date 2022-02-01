import numpy as np

def rotate(coords, angle=0, axis=2):
    """
        Clockwise rotation
    """
    if angle==0:
        return coords

    # reshape
    original_shape = coords.shape

    coords = np.reshape(coords, [-1, 3])

    # Prepare to rotate the coordinates   
    rad = np.radians(angle)
    c, s = np.cos(rad), np.sin(rad)

    # Prepare the rotation matrix
    # print('rotation axis', axis)
    if axis == 0:
        # Tilt
        # image = ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(0,1), reshape=False)
        rotation_matrix = np.array([
                [1,  0,  0 ],
                [0,  c,  s],
                [0, -s,  c]])
                
    elif axis == 1:
        # Tilt
        # image = ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(0,2), reshape=False)
        rotation_matrix = np.array([
                [c,  0, s],
                [0,  1, 0],
                [-s, 0, c]])
                
    elif axis == 2:
        # In this case, it rotates on the "main" axis
        # image = ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(1,2), reshape=False)

        rotation_matrix = np.array([
                [c,  s,  0 ],
                [-s, c,  0 ],
                [0,  0,  1.]])
    
    # Do the rotation on point (0,0,0)
    coords = np.dot(rotation_matrix, coords.T).T
    
    coords = np.reshape(coords, original_shape)
    return coords