from scipy.spatial import Delaunay, KDTree
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sc
from icecream import ic
import time
import config

from wssnet.utility import h5util
import ObjHandler as obc
from CFDResult import CFDResult
import data_util as du

if __name__ == "__main__":
    mesh_file = f'{config.ROOT_DIR}/examples/example_aorta_reg.obj'
    csv_dir   = f'{config.ROOT_DIR}/examples/csv'
    
    output_filepath = f'{csv_dir}/test_sheet.h5'

    # file prefix
    prefix = 'export'
    
    # start, end and step (depending on your export files)
    t_start = 10
    t_end = 30
    t_step = 10

    # filename format
    csv_wall_file = f'{prefix}_wall-{t_start:04}'
    wall_file = f'{csv_dir}/{csv_wall_file}'

    # decimal rounding digit
    rounding = 4
    
    # KD-tree parameters (for wall nodes)
    radius = 7 # search radius (in mm)
    k_neighbor = 1
    mask_dist_threshold = 1.2 # mm (below this threshold considered as wall)

    probe_dist = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0])
    # min probe radius
    min_probe_radius = 0.3
    # get all the sheets above the min probe radius    
    probe_dist = probe_dist[probe_dist >= min_probe_radius]
    ic(probe_dist)

    # 1. Load the mesh
    mesh = obc.ObjHandler(mesh_file, rounding)
    mesh.stats()
    v_idx = mesh.get_sorted_uv_based_vertex_index()
    
    sorted_vertex = mesh.vertices[v_idx]
    sorted_normals = mesh.normals[v_idx]
    
    xyz = du.get_flatmap(mesh, sorted_vertex)
    normals_grid = du.get_flatmap(mesh, sorted_normals)

    # save the wall coordinates and point normals
    h5util.save_to_h5(output_filepath, "xyz0", xyz)
    h5util.save_to_h5(output_filepath, "normals", normals_grid)
    
    # Build a KD tree for the wall points
    # Load the velocity data
    vel_cfd_res = CFDResult(wall_file, rounding, True, False)

    # Reshape xyz coordinates
    coords = np.stack((vel_cfd_res.x, vel_cfd_res.y, vel_cfd_res.z), axis=-1)

    n_points = len(coords)
    tree = KDTree(coords, leafsize=10)
    
    # Prepare probe points for wall
    probe1 = sorted_vertex
    ic(probe1.shape)
    # Query the KD tree 
    # when not found, this return an index==nr_of_points
    distances, ndx = tree.query(probe1, k=k_neighbor, distance_upper_bound=radius)
    dist_grid = mesh.fill_grid(distances)
    wss_mask  = dist_grid <= mask_dist_threshold
    h5util.save_to_h5(output_filepath, "wss_mask", wss_mask)
    
    # ======== Preload and triangulate ======== 
    velocity_file = f'{csv_dir}/{prefix}-{t_start:04}'
    vel_cfd_res = CFDResult(velocity_file, rounding, False, False)
    # Reshape xyz coordinates
    v_coords = np.stack((vel_cfd_res.x, vel_cfd_res.y, vel_cfd_res.z), axis=-1)

    # Triangulate once
    # https://stackoverflow.com/questions/51858194/storing-the-weights-used-by-scipy-griddata-for-re-use/51937990
    # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    start_time = time.time()
    tri = Delaunay(v_coords)  # Compute the triangulation once for massive speedup
    print(f"Delaunay triangulation: {(time.time()-start_time):.1f} sec")

    # ======== iterate through time frames ===============
    row_index = 0
    first_row = True
    for i in range(t_start,t_end, t_step):    
        print(f"Processing time {i} ...")
        start_time = time.time()

        csv_file = f'{csv_dir}/{prefix}_wall-{i:04}'
        velocity_file = f'{csv_dir}/{prefix}-{i:04}'

        # Get the wall coordinates and WSS
        wall_cfd_res = CFDResult(csv_file, rounding, True, False)

        # Fill in to grid
        closest_wss = du.get_wss_from_nearest_coords(ndx, wall_cfd_res.wss)
        wss_grid = mesh.fill_grid(closest_wss)
        wss_grid = np.nan_to_num(wss_grid)
        
        h5util.save_to_h5(output_filepath, "wss_clean", wss_grid)
        # show the first
        if first_row:
            first_row = False
            # show this for the first time
            plt.imshow(wss_grid, cmap='jet', clim=[0,40]) #, clim=[0, 30])
            plt.colorbar()
            plt.show()
        
        # --- wss x
        wss_grid = du.get_wss_from_nearest_coords(ndx, wall_cfd_res.wssx)
        wssx = mesh.fill_grid(wss_grid)
        
        # --- wss y
        wss_grid = du.get_wss_from_nearest_coords(ndx, wall_cfd_res.wssy)
        wssy = mesh.fill_grid(wss_grid)
        
        # --- wss z
        wss_grid = du.get_wss_from_nearest_coords(ndx, wall_cfd_res.wssz)
        wssz = mesh.fill_grid(wss_grid)
                
        # stack them to 3 channel
        wss_vector = np.stack((wssx,wssy,wssz), axis=-1)
        wss_vector = np.nan_to_num(wss_vector)
        
        # save the wss vector
        h5util.save_to_h5(output_filepath, "wss_vector", wss_vector)


        # ================== Velocity Data ==================
        # Load the velocity data
        vel_cfd_res = CFDResult(velocity_file, rounding, False, True)

        # Prepare interpolator
        print("Preparing velocity interpolation")
        interpolator_vx = sc.LinearNDInterpolator(tri, vel_cfd_res.vx)
        interpolator_vy = sc.LinearNDInterpolator(tri, vel_cfd_res.vy)
        interpolator_vz = sc.LinearNDInterpolator(tri, vel_cfd_res.vz)

        # Prepare probe points for velocity
        for j, dist in enumerate(probe_dist):
            # Probe velocity at several inward normals
            print(f"Probing {dist}mm ...")

            probe1 = sorted_vertex - (dist * sorted_normals)
            
            if (row_index==0):
                # Save the inner coordinates once (when reading first row)
                xyz = du.get_flatmap(mesh, probe1)                
                h5util.save_to_h5(output_filepath, f"xyz{dist}", xyz)
            
            # Probe the velocity values
            vx1 = interpolator_vx(probe1)
            vy1 = interpolator_vy(probe1)
            vz1 = interpolator_vz(probe1)

            # # Fill in to grid
            vx_grid = mesh.fill_grid(vx1)
            vy_grid = mesh.fill_grid(vy1)
            vz_grid = mesh.fill_grid(vz1)

            speed_grid = np.stack((vx_grid,vy_grid,vz_grid), axis=-1)
            speed_grid = np.nan_to_num(speed_grid)
            # ic(speed_grid.shape)
            
            h5util.save_to_h5(output_filepath, f"v{dist}", speed_grid)
        # end of probing loop
        row_index += 1 # this is for the distance dt..which is only saved once
        
        print(f"Elapsed: {(time.time()-start_time):.1f} sec")
    print(f'Done case!')
