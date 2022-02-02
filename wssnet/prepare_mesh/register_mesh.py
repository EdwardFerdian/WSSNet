import time
import os
import pyvista as pv
import numpy as np
from icecream import ic
import pickle
import config
from wssnet.pycpd import RigidRegistration, AffineRegistration, DeformableRegistration
from wssnet.prepare_data import ObjHandler as  obc
import config

def read_obj_file(filename, vertex_only):
    lines = open(filename).readlines()

    # ic(len(lines))
    # ic(lines[5:6])
    vertices = []
    for i in range(0, len(lines)):
        if lines[i].startswith('v '):
            line = lines[i]
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)

            vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]

            vertices.append(vertex) # remove the 'v'
        elif lines[i].startswith('vt '):
            break
    
    # ic(len(vertices), vertices[0:2])
    vertices = np.asarray(vertices)
    if vertex_only:
        return vertices, None
    else:
        # copy the rest
        lines_to_copy = lines[i:]
        return vertices, lines_to_copy

def register_rigid(X, Y):
    start_time = time.time()

    reg = RigidRegistration(**{'X': X, 'Y': Y, 'max_iterations': max_iter})
    Z, params_rigid = reg.register()
    # ic(params_rigid, Z[0:2])
    
    elapsed = time.time()-start_time
    print(f'\nRigid reg done {elapsed:.1f} sec.\n')

    save_params(output_dir, 'params_rigid', params_rigid)
    return Z

def register_affine(X, Y):
    start_time = time.time()

    aff = AffineRegistration(**{'X': X, 'Y': Y, 'max_iterations': max_iter, 'tolerance': tolerance})
    Z, params_affine = aff.register()

    elapsed = time.time()-start_time
    print(f'\nAffine reg done {elapsed:.1f} sec.\n')
    save_params(output_dir, 'params_affine', params_affine)
    return Z

def register_deform(X, Y, alpha, beta, tolerance, save_par=False):
    start_time = time.time()

    defo = DeformableRegistration(**{'X': X, 'Y': Y, 'max_iterations': max_iter, 'alpha': alpha, 'beta': beta, 'tolerance': tolerance})
    Z, params_deform = defo.register()
    
    elapsed = time.time()-start_time
    print(f'\nDeform reg done {elapsed:.1f} sec.\n')
    
    if save_par:
        save_params(output_dir, 'params_deform', params_deform)
    return Z

def register_aorta(source_vertices, target_vertices, alpha, beta):
    """
        Apply rigid, affine, and deformable registration
    """
    X = target_vertices
    Y_arr = source_vertices

    # ic(Y_arr.shape, Y_arr[0:2])
    # ic(X[0:2])

    # -- do registration --
    Z = register_rigid(X, Y_arr)
    A = register_affine(X, Z)
    result = register_deform(X, A, alpha, beta, 1e-3, True)

    return result

def save_params(output_dir, output_name, var_to_save):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(f'{output_dir}/{output_name}.pkl', 'wb') as f:
        pickle.dump(var_to_save, f)

def save_result(output_dir, output_name, content):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f'{output_dir}/{output_name}'

    ic(content, content.shape)
    np.savetxt(filename, content, fmt="v %.6f %.6f %.6f")

def save_to_obj(output_dir, output_name, content, lines_to_copy):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f'{output_dir}/{output_name}'


    with open(filename, 'w') as f:
        # write headers
        f.write("# Blender v2.91.2 OBJ File\n")
        f.write(f"# Registered mesh - CPD rigid affine deform\n")
        f.write(f"o {output_name}\n")

    with open(filename, 'a') as f:
        np.savetxt(f, content, fmt="v %.6f %.6f %.6f")
        f.writelines(lines_to_copy)

def prepare_faces(faces):
    # ic(faces, faces.shape)
    faces = faces.astype(int)
    faces = faces - 1 # turn to 0 index
    # we need to insert the number of points in a face (vtk face format)
    faces = np.insert(faces,0,4,axis=1)
    faces = np.hstack(faces)
    return faces

def inflate_mesh(vertices, inflation_dist = 0.5):
    print(f"Inflate mesh")
    mesh = obc.ObjHandler(template_fine_file, 6)

    # Prepare the face indexes from the fine template
    faces = prepare_faces(mesh.faces)

    # Construct surface pyVista
    surf = pv.PolyData(vertices, faces)
    
    # Recalculate point normals
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True)

    # inflate the object using the normals 
    new_vertices = vertices + inflation_dist * surf.point_normals
    return new_vertices

def save_mesh_using_template(output_name, new_vertices, header):
    # Load a template of the fine mesh
    mesh = obc.ObjHandler(template_fine_file, 6)

    # Prepare the face indexes from the fine template
    faces = prepare_faces(mesh.faces)

    # Construct surface pyVista
    surf = pv.PolyData(new_vertices, faces)
    
    # Recalculate point normals
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True)

    # ==== Save the new inflated mesh ====
    # replace the template's vertice and normals with the new one
    mesh.vertices = new_vertices 
    mesh.normals = surf.point_normals
    mesh.save_to_obj(output_dir, output_name, header)

    return surf

def show_plot(new_surf, target_surf):
    p = pv.Plotter(border=True)
    p.add_mesh(new_surf, opacity=0.5, show_edges=True, label="Registered", color="white")
    p.add_mesh(target_surf, opacity=0.2, label="Target", color="blue")
    p.add_legend()
    
    arrows = new_surf.glyph(orient='Normals', scale=False, factor=1) # , tolerance=0.05)
    p.add_mesh(arrows, color="black") # arrows are the computed normals from pyvista
    
    p.show_bounds(all_edges=True)

    p.show()
    
if __name__ == '__main__':
    
    case_name = f'example_aorta'
    target_file   = rf'{config.ROOT_DIR}/examples/{case_name}.obj'

    # registration iteration
    max_iter = 1000
    # deformable params https://github.com/siavashk/pycpd/issues/27
    # Basically ⍺ is how much regularization to apply. 
    # If it is too high, the point cloud acts rigid and if it is too low it acts fluid. 
    # β is the neighbourhood of points to consider when applying the rigidity constraint.
    alpha = 3
    beta  = 15
    tolerance = 1e-4 # tolerance for affine and second deform

    alpha2 = 3
    beta2  = 7
    
    # Mesh template files
    template_file = f'{config.TEMPLATE_DIR}/aorta_template12.obj'
    template_fine_file = f'{config.TEMPLATE_DIR}/aorta_template48.obj'
    subdiv_matrix_file = f"{config.TEMPLATE_DIR}/subdiv_matrix_12_48.txt"

    # Input output dirs
    output_name = f'{case_name}_reg_coarse_ab{alpha}{beta}.obj'
    subdiv_output_name = f'{case_name}_subdiv.obj'
    final_output_name  = f'{case_name}_reg.obj'
    output_dir = f'{config.MESH_DIR}/reg_{case_name}'
    
    # ==== Read source and target mesh ====
    source_vertices, lines_to_copy = read_obj_file(template_file, False)
    target_vertices, _ = read_obj_file(target_file, True)
    
    # ==== Performing registration ====
    # Perform registration of the coarse mesh
    result = register_aorta(source_vertices, target_vertices, alpha, beta)

    # save the registration of coarse mesh
    save_to_obj(output_dir, output_name, result, lines_to_copy)


    # ==== Apply subdivision surface ====
    # load the subdivision matrix
    subdiv_matrix = np.loadtxt(subdiv_matrix_file)

    # calculate the subdiv surface
    new_vertices = np.matmul(subdiv_matrix, result)

    # Inflate the mesh
    new_vertices = inflate_mesh(new_vertices)

    # ==== Save the subdivided registered mesh ====
    # obj header
    header = f'Registered coarse mesh - alpha {alpha} beta {beta}'
    new_surf = save_mesh_using_template(subdiv_output_name, new_vertices, header)


    # ==== VIsualization ====
    # prepare target mesh in pyVista
    target_surf = pv.read(target_file)
    # original template
    temp = pv.read(template_file)

    # Visualize the mesh so we don't need to go to Blender :)
    show_plot(new_surf, target_surf)

        
    # ==== Peform 1 more deformable registration on the subdived mesh ====
    ic(target_vertices.shape, new_vertices.shape)
    new_vertices = np.asarray(new_vertices)
    res = register_deform(target_vertices, new_vertices, alpha2, beta2, tolerance)

   

    # ==== Save the final registration ====
    header = f'Registered subdivided mesh - alpha {alpha2} beta {beta2}'
    new_surf = save_mesh_using_template(final_output_name, res, header)

    # Visualize
    show_plot(new_surf, target_surf)

