from scipy.spatial import Delaunay, KDTree
import numpy as np
import scipy.interpolate as sc
from icecream import ic
import time
import config

from wssnet.utility import h5util
import ObjHandler as obc
from CFDResult import CFDResult

def get_minmax_arr(arr, skip_x):
    min_x, max_x = np.min(arr), np.max(arr)
    min_x, max_x = np.floor(min_x), np.ceil(max_x)
    
    min_x = min_x - skip_x
    max_x = max_x + skip_x
    x_arr = np.arange(min_x, max_x , skip_x)
    return x_arr, min_x, max_x
    

if __name__ == "__main__":

    mesh_file = f'{config.ROOT_DIR}/examples/example_aorta_reg.obj'
    csv_dir   = f'{config.ROOT_DIR}/examples/csv'

    output_filepath = f'{csv_dir}/test_grid.h5'
    
    # file prefix
    prefix = 'export'
    
    # start, end and step (depending on your export files)
    t_start = 10
    t_end = 30
    t_step = 10

    # decimal rounding digit
    rounding = 4
    # modify this dx param for different voxel size
    dx = 2.4

    # the "origin" is calculated based on base-resolution
    base_dx = 2.4
    

    # 1. Load the mesh
    mesh = obc.ObjHandler(mesh_file, rounding)
    mesh.stats()
    v_idx = mesh.get_sorted_uv_based_vertex_index()
    
    sorted_vertex = mesh.vertices[v_idx]
    sorted_normals = mesh.normals[v_idx]

    # ================== Velocity Data ==================
    velocity_file = f'{csv_dir}/export-{t_start:04}'

    # Load the velocity data
    vel_cfd_res = CFDResult(velocity_file, rounding, False, True)

    # Reshape xyz coordinates
    v_coords = np.stack((vel_cfd_res.x, vel_cfd_res.y, vel_cfd_res.z), axis=-1)
    
    x_arr, min_x, max_x = get_minmax_arr(v_coords[:,0], base_dx)
    y_arr, min_y, max_y = get_minmax_arr(v_coords[:,1], base_dx)
    z_arr, min_z, max_z = get_minmax_arr(v_coords[:,2], base_dx)

    xx, yy, zz = np.mgrid[min_x:max_x: dx, min_y:max_y:dx, min_z:max_z:dx]
    ic(len(yy), len(zz))
    yy = np.asarray(yy)
    zz = np.asarray(zz)
    ic(xx.shape, yy.shape, zz.shape)

    # --- get mask ---
    tree = KDTree(v_coords, leafsize=10)
    probe1 = np.stack((xx, yy, zz), axis=-1)
    distances, ndx = tree.query(probe1, k=1, distance_upper_bound=dx)
    print(distances.shape)

    # minimum distance is 0.4 because we have our first inflation layer 0.3mm on CFD
    mask = distances <= np.max([(dx / 2), 0.4])
    
    h5util.save_to_h5(output_filepath, f"dx", (dx,dx,dx))
    h5util.save_to_h5(output_filepath, f"origin", (min_x,min_y,min_z))
    h5util.save_to_h5(output_filepath, f"distance_mask", distances)
    h5util.save_to_h5(output_filepath, f"mask", mask)
    
    # Triangulate once
    # https://stackoverflow.com/questions/51858194/storing-the-weights-used-by-scipy-griddata-for-re-use/51937990
    # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    start_time = time.time()
    tri = Delaunay(v_coords)  # Compute the triangulation once for massive speedup
    print(f"Delaunay triangulation: {(time.time()-start_time):.1f} sec")

    # ======== iterate through time frames ===============
    for i in range(t_start, t_end, t_step):
        print(f"Processing time {i} / {t_end}")
        start_time = time.time()
        velocity_file = f'{csv_dir}/export-{i:04}'
        # Load the velocity data
        vel_cfd_res = CFDResult(velocity_file, rounding, False, True)

        # Prepare interpolator
        print("Preparing velocity interpolation")
        interpolator_vx = sc.LinearNDInterpolator(tri, vel_cfd_res.vx)
        interpolator_vy = sc.LinearNDInterpolator(tri, vel_cfd_res.vy)
        interpolator_vz = sc.LinearNDInterpolator(tri, vel_cfd_res.vz)

        print("Interpolating...")
        vx1 = interpolator_vx(xx,yy,zz)
        vy1 = interpolator_vy(xx,yy,zz)
        vz1 = interpolator_vz(xx,yy,zz)

        ic(vx1.shape)
        vx1 = np.nan_to_num(vx1)
        vy1 = np.nan_to_num(vy1)
        vz1 = np.nan_to_num(vz1)

        h5util.save_to_h5(output_filepath, f"u", vx1)
        h5util.save_to_h5(output_filepath, f"v", vy1)
        h5util.save_to_h5(output_filepath, f"w", vz1)
        print(f"Elapsed: {(time.time()-start_time):.1f} sec")
    print(f'Done!')
    