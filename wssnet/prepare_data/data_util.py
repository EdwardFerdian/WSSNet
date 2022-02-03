import numpy as np
import skimage
from skimage import morphology
import scipy.interpolate as sc

def get_flatmap(mesh, xyz_values):
    """
        Mesh is an obj handler, xyz values are coordinates (n,3)
    """
     # save the wall coords
    x_grid = mesh.fill_grid(xyz_values[:,0])
    y_grid = mesh.fill_grid(xyz_values[:,1])
    z_grid = mesh.fill_grid(xyz_values[:,2])

    xyz = np.stack((x_grid,y_grid,z_grid), axis=-1)
    return xyz


def get_crack(img, use_median):
    """
        Due to some reasons, some CFD results appear to have 'cracks' on the velocity field
        This method is to detect the crack within the flatmap using lapacian filter
    """
    # get the "edge"
    mask = skimage.filters.laplace(img)
    mask = np.abs(mask) #remove the sign
    
    if use_median:
        # get a binary mask
        mask = mask > np.median(mask) 
    else:
        mask = mask > np.mean(mask) 
    # erode the thick edge to get a single crack line
    mask = morphology.binary_erosion(mask) 

    # trim the sides (we know there are no cracks from the sides)
    # mask[:, :5] = 0
    # mask[:, -5:] = 0
    return mask * 1

def remove_cracks(img, crack):
    """
        Smoothing the 'cracks' by applying interpolated values.
        It is not perfect, but it does the job pretty well.
    """
    # -------  get cracks ------- 
    mask = crack

    # get the area containing the values
    valid_mask = mask == 0

    # ------- prepare for interpolation ------- 
    coords = np.array(np.nonzero(valid_mask)).T
    value_list = img[valid_mask]

    it = sc.LinearNDInterpolator(coords, value_list)
    interpolated = it(list(np.ndindex(img.shape))).reshape(img.shape)

    # NaN checking, just in case
    # if the interpolated happens in the edge, it will result in nan
    # in that case, replace the nans, with the original value from img

    interpolated = np.where(np.isnan(interpolated), img, interpolated)
    # ic (interpolated)

    # ------- merge image
    # we only use interpolated value on the crack (mask)
    return img * valid_mask + interpolated * mask