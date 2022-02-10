import pyvista as pv
import h5py
import numpy as np
import config
import os
import wssnet.utility.tawss_utils as util


if __name__ == '__main__':
    wss_color = 'inferno'
    osi_color = 'viridis'
    
    dt = 4e-2 # time in seconds

    hd5path = f"{config.ROOT_DIR}/examples/case70_prediction.h5"

    
    if not os.path.exists(hd5path):
        print(f'{hd5path} does not exists!')
    else:
        # load wss prediction and wall coordinates
        with h5py.File(hd5path, 'r') as hf:
            wss_pred = np.asarray(hf.get('wss'))
            coords = hf.get('xyz0')[0]

        # calculate osi
        osi_pred = util.get_osi(wss_pred, dt)
        
        # calculate tawss
        wss_val_pred = util.get_wss_magnitude(wss_pred)
        tawss_pred = util.get_tawss(wss_val_pred, dt)
        
        tawss = np.reshape(tawss_pred, [-1])
        osi = np.reshape(osi_pred, [-1])

        surf_wss = pv.PolyData(coords)
        surf_wss["wss"] = tawss

        surf_osi = pv.PolyData(coords)
        surf_osi["osi"] = osi

        # white background
        # pv.set_plot_theme("document")
        maxval = 6

        p = pv.Plotter(notebook=0, shape=(1, 2), border=False)
        p.subplot(0, 0)
        p.add_text("TAWSS")
        p.add_mesh(surf_wss, scalars="wss", cmap='inferno', clim=[0,maxval])

        p.subplot(0, 1)
        p.add_text("OSI")
        p.add_mesh(surf_osi, scalars="osi", cmap='viridis', clim=[0,0.5])

        
        p.view_isometric()
        p.link_views()
        p.show()