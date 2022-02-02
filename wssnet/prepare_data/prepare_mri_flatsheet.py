import numpy as np
import ObjHandler as obc
import h5py
import pyvista as pv
import config
import os
from wssnet.utility import h5util, wss_utils, augmentation



def load_vector_fields(input_filepath, columns, idx):
    with h5py.File(input_filepath, 'r') as hf:
        u = np.asarray(hf.get(columns[0])[idx])
        v = np.asarray(hf.get(columns[1])[idx])
        w = np.asarray(hf.get(columns[2])[idx])
        dx = np.asarray(hf.get('dx')[0])
        if 'origin' in hf:
            origin = np.asarray(hf.get('origin')[0])
        else:
            origin = None
    return u, v, w, dx, origin

def get_mapping_index(surf, rotated_vertex):
    surf_index_map = []
    for point_to_find in rotated_vertex:
        # returns a tuple of index
        found = np.where((surf.points[:,0] == point_to_find[0]) & (surf.points[:,1] == point_to_find[1]) & (surf.points[:,2] == point_to_find[2]))

        if len(found[0]) == 0:
            # print('not found')
            surf_index_map.append(np.nan)
        else:
            # keep the first index only
            surf_index_map.append(found[0][0])
        # ic(found, point_to_find, len(found[0]))
        # ic(surf.points[found])
    return surf_index_map

def get_flatmap(mesh, xyz_values):
     # save the wall coords
    x_grid = mesh.fill_grid(xyz_values[:,0])
    y_grid = mesh.fill_grid(xyz_values[:,1])
    z_grid = mesh.fill_grid(xyz_values[:,2])

    xyz = np.stack((x_grid,y_grid,z_grid), axis=-1)
    return xyz



def convert_grid_to_sheet(std_noise=0):
    """
        Extract the surface coordinates from the registered mesh
        Extract the velocity vectors based on the registered mesh

        Convert the extracted points into the flatmap/sheets representation
    """
    # 1. Load the mesh
    mesh = obc.ObjHandler(mesh_filename, rounding)
    mesh.stats()
    v_idx = mesh.get_sorted_uv_based_vertex_index()
    
    sorted_vertex  = mesh.vertices[v_idx]
    sorted_normals = mesh.normals[v_idx]

    # ic(sorted_vertex)
    if is_actual_mri:
        # Using ITK-SNAP, the axis is sometimes swapped, so we need to rotate
        sorted_vertex = augmentation.rotate(sorted_vertex, 90, 2)
        sorted_normals = augmentation.rotate(sorted_normals, 90, 2)

    inward_normals =  sorted_normals * -1
    # probe1 (probe based on surface points)
    probe1 = sorted_vertex - (dists[0] * sorted_normals)
    probe2 = sorted_vertex - (dists[1] * sorted_normals)

    # ----- READY TO SAVE ---
    xyz0 = get_flatmap(mesh, sorted_vertex)
    xyz1 = get_flatmap(mesh, probe1)
    xyz2 = get_flatmap(mesh, probe2)
    normals_grid = get_flatmap(mesh, sorted_normals)

    h5util.save_to_h5(output_filepath, "xyz0", xyz0)
    h5util.save_to_h5(output_filepath, f"xyz{dists[0]}", xyz1)
    h5util.save_to_h5(output_filepath, f"xyz{dists[1]}", xyz2)
    h5util.save_to_h5(output_filepath, f"normals", normals_grid)


    # # prepare the polydata probes
    pc1 = pv.PolyData(probe1)
    pc2 = pv.PolyData(probe2)
    pcs = [pc1, pc2]

    # Load velocity file
    columns = ['u', 'v', 'w']
    
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        len_indexes = len(hdf5[columns[0]])


    for row_index in range(0, len_indexes):
        print(f"\rProcessing row {row_index+1}/{len_indexes}", end="")
        u,v,w, dx, origin = load_vector_fields(input_filepath, columns, row_index)

        if add_noise:
            # print(f"Adding noise {noise_level}% {std_noise}m/s")
            noise = np.random.normal(0, std_noise, u.shape)
            u += noise

            noise = np.random.normal(0, std_noise, u.shape)
            v  += noise

            noise = np.random.normal(0, std_noise, u.shape)
            w  += noise

        velocity = wss_utils.create_uniform_vector(u,v,w, dx)
        if not is_actual_mri:
            velocity.origin = origin 


        # probe the velocity
        pc_tangents = []

        # no slip condition (to calculate the WSS, we need to insert 0 velocity at wall)
        pc_tangents.append(np.zeros(len(pc1.points)))

        for i in range(0, nr_points):

            pc = pcs[i].sample(velocity)
            vector1 = wss_utils.extract_vectors(pc)

            v1 = get_flatmap(mesh, vector1)
            h5util.save_to_h5(output_filepath, f"v{dists[i]}", v1)
            

            # get the tangential vector
            pc1_normals, pc1_tangent = wss_utils.get_orthogonal_vectors(vector1, inward_normals)
            pc1_tangent_mag = wss_utils.get_vector_magnitude(pc1_tangent)
            
            # optional: save tangential velocity (magnitude)
            # vtan = get_flatmap(pc1_tangent_mag)
            # h5util.save_to_h5(output_filepath, f"vtan{dists[i]}", vtan)

            pc_tangents.append(pc1_tangent_mag)
        
        # calculate WSR and WSS
        parabolic_fitting = True
        gradients = wss_utils.calculate_gradient(pc_tangents, dists, use_parabolic=parabolic_fitting)
        
        # 9. Assign the gradients back to the surface
        wss = gradients * viscosity
        wss_grid = mesh.fill_grid(wss)

        # The |wss| is assumed to have the same direction as v1_tan
        # which means we can use v_tan with a certain multiplier to get there
        multiplier = wss / pc1_tangent_mag
        multiplier = np.nan_to_num(multiplier)
        wss_vector = pc1_tangent * multiplier[:,np.newaxis]

        # wss vector is assumed to have the same direction as v1
        wss_vector = get_flatmap(mesh, wss_vector)

        # ---- SAVE ---
        h5util.save_to_h5(output_filepath, "wss_clean", wss_grid) # |wss|
        h5util.save_to_h5(output_filepath, "wss_vector", wss_vector) # wss vector
    # as we don't have the masking for now, mark them all as valid regions
    h5util.save_to_h5(output_filepath, "wss_mask", np.ones(wss_grid.shape))


if __name__ == "__main__": 
    # ========== Input/output filename ========== 

    # Input: Registered surface template
    mesh_filename = f'{config.ROOT_DIR}/examples/example_aorta_reg.obj'
    # Input: Image grid file (velocity grid)
    input_filepath = f'{config.ROOT_DIR}/examples/example_grid.h5'

    # Output: Coordinates and velocity sheets
    output_filepath = f'{config.ROOT_DIR}/examples/example_sheet.h5'

    # With actual MRI, if segmentation is performed on ITK-SNAP, the axis is rotated
    # Turn on this  option is to swap the axis back
    # For CFD data, keep this off
    is_actual_mri = False

    # Add noise to the sheets, use for CFD data only (for augmentation)
    add_noise = False

    # Params to calculate WSS using parabolic fitting method
    viscosity = 4 # in cPa unit
    rounding = 6
    threshold = 1e-3
    nr_points = 2
    
    # Probing distance to build the sheets
    dists = [1.0, 2.0] # in mm

    if not os.path.exists(mesh_filename):
        print(f'{mesh_filename} does not exists')

    if not os.path.exists(input_filepath):
        print(f'{input_filepath} does not exists')
    
    std_noise = 0
    if add_noise:
        noise_level = 2 # in percent
        venc = 1.5 # max venc in m/s
        std_noise = venc * noise_level  / 100 # std noise in m/s
        print(f"Adding noise {noise_level}% {std_noise}m/s")
    
    print(f"Preparing sheets...")
    convert_grid_to_sheet(std_noise)