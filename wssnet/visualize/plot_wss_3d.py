import pyvista as pv
import numpy as np
from wssnet.prepare_data import ObjHandler as  obc
import config
import h5py


def prepare_faces(faces):
    # ic(faces, faces.shape)
    faces = faces.astype(int)
    faces = faces - 1 # turn to 0 index
    # we need to insert the number of points in a face (vtk face format)
    faces = np.insert(faces,0,4,axis=1)
    faces = np.hstack(faces)
    return faces

def load_img(hd5path, row_idx, trim_root=False):
    with h5py.File(hd5path, 'r') as hl:
        coords = hl.get('xyz0')[0]        
        wss = hl.get('wss')[row_idx]
        wss_gt = hl.get('wss_true')[row_idx]

    if trim_root:
        coords = coords[:,2:]
        wss = wss[:,2:]
        wss_gt = wss_gt[:,2:]
        
    return coords, wss, wss_gt

def get_magnitude(wss_vector):
    if (wss_vector.shape[-1] != 3):
        return wss_vector

    return np.sum(wss_vector ** 2, axis=-1) ** 0.5

def create_mesh(idx):
    coords, wss, wss_gt = load_img(hd5path, idx, trim_root)
    
    wss = get_magnitude(wss)
    wss_gt = get_magnitude(wss_gt)

    surf["wss"] = np.reshape(wss, [-1])
    surf_gt["wss"] = np.reshape(wss_gt, [-1])

    maxVal = np.max(wss)
    maxVal = 8
    p.subplot(0, 0)
    p.add_text("WSS prediction")
    p.add_mesh(surf, scalars="wss", cmap='inferno', clim=[0,maxVal])

    p.subplot(0, 1)
    p.add_text(f"WSS Ground Truth")
    p.add_mesh(surf_gt, scalars="wss", cmap='inferno', clim=[0,maxVal])

    p.link_views()



if __name__ == '__main__':
    hd5path = f"{config.ROOT_DIR}/examples/case70_prediction.h5"
    trim_root = True

    template_fine_file = f'{config.TEMPLATE_DIR}/aorta_template48.obj'
    mesh = obc.ObjHandler(template_fine_file, 6)

    # # Prepare the face indexes from the fine template
    faces = prepare_faces(mesh.faces)

    coords, wss, wss_gt = load_img(hd5path, 0, trim_root)
    coords = np.reshape(coords, [-1, 3])

    surf = pv.PolyData(coords)
    surf_gt = pv.PolyData(coords)

    # pv.set_plot_theme("dark")
    p = pv.Plotter(notebook=0, shape=(1, 2), border=False)

    # Generate the point clouds here    
    p.add_slider_widget(create_mesh, [0, 72], title='Timeframe')

    p.view_isometric()
    p.show(full_screen=False)
