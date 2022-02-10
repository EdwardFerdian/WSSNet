import matplotlib.pyplot as plt
import h5py
import numpy as np
import config
from skimage.metrics import structural_similarity as ssim
import os
import wssnet.utility.tawss_utils as util


def pearson(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    diff_x = (x - mean_x)
    diff_y = (y - mean_y)
    
    nom   = np.sum( diff_x * diff_y)
    denom = np.sqrt(np.sum(diff_x ** 2) * np.sum(diff_y ** 2))

    r = nom / denom
    return r

if __name__ == '__main__':
    save_img = False
    
    wss_color = 'inferno'
    osi_color = 'viridis'
    
    print('MAE std rel ssim ssim_norm pearson')
    dt = 4e-2 # time in seconds

    img_dir = f'{config.ROOT_DIR}/images'

    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    
    hd5path = f"{config.ROOT_DIR}/examples/example_prediction.h5"

    
    if not os.path.exists(hd5path):
        print(f'{hd5path} does not exists!')
    else:
        # load wss prediction and wss gt
        with h5py.File(hd5path, 'r') as hf:
            wss_pred = np.asarray(hf.get('wss'))
            wss_true = np.asarray(hf.get('wss_true'))

        # calculate osi
        osi_pred = util.get_osi(wss_pred, dt)
        osi_true = util.get_osi(wss_true, dt)

        # calculate tawss
        wss_val_pred = util.get_wss_magnitude(wss_pred)
        wss_val_true = util.get_wss_magnitude(wss_true)

        tawss_pred = util.get_tawss(wss_val_pred, dt)
        tawss_gt = util.get_tawss(wss_val_true, dt)

        # get tawss diff and relative diff
        eps = 0.05
        tawss_diff = np.abs(tawss_pred - tawss_gt)
        tawss_rel  = 100 * tawss_diff / (tawss_gt + eps)

        # calculate ssim
        ssim_noise = ssim(tawss_gt, tawss_pred,
                    data_range=tawss_gt.max() - tawss_gt.min())

        # calculate pearson correlation
        r = pearson(tawss_pred, tawss_gt)

        tawss_gt_norm = tawss_gt / np.max(tawss_gt)
        tawss_pred_norm = tawss_pred / np.max(tawss_pred)
        ssim_norm =  ssim(tawss_gt_norm, tawss_pred_norm,
                    data_range=1)

        # print stats        
        print(np.mean(tawss_diff), np.std(tawss_diff), np.mean(tawss_rel), ssim_noise, ssim_norm, r)

        # show image
        plt.subplots_adjust(left=.01, bottom=0.01, right=0.99, top=.95, wspace=0.01, hspace=0.05)

        maxval = 12
        plt.subplot(221), plt.imshow(tawss_pred, cmap=wss_color, clim=[0, maxval])
        plt.title(f'TAWSS ({r:.2f})')
        plt.xticks([]), plt.yticks([])
        plt.colorbar()

        maxval = 4
        plt.subplot(222), plt.imshow(tawss_gt, cmap=wss_color, clim=[0,maxval])
        plt.title('TAWSS GT')
        plt.xticks([]), plt.yticks([])
        plt.colorbar()

        plt.subplot(223), plt.imshow(osi_pred, cmap=osi_color, clim=[0, 0.5])
        plt.title('OSI pred')
        plt.xticks([]), plt.yticks([])
        plt.colorbar()

        plt.subplot(224), plt.imshow(osi_true, cmap=osi_color, clim=[0, 0.5])
        plt.title('OSI GT')
        plt.xticks([]), plt.yticks([])
        plt.colorbar()

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        if save_img:
            plt.savefig(f'{img_dir}/tawss_osi.png', dpi=200)
            plt.close()
        else:
            
            plt.show()