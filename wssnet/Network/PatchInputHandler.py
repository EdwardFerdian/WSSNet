import numpy as np
import h5py
import matplotlib.pyplot as plt
from random import randrange

class PatchInputHandler():
    # constructor
    def __init__(self, data_filename, scale_dist, patch_size, distances, use_avg_patch):
        self.patch_size = patch_size
        self.data_filename = data_filename
        self.scale_dist = scale_dist

        # if true, we crop by 2 pixels starting from second patch to avoid patch border artefact
        self.use_avg_patch = use_avg_patch
        self.overlap_size = 2

        self.distances = distances
        
        self.coord_colnames = []
        self.velocity_colnames = []
        
        self.wall_coord  = 'xyz0'
        self.wss_colname = 'wss_vector'
        
        for dist in self.distances:
            self.coord_colnames.append(f'xyz{dist}')
            self.velocity_colnames.append(f'v{dist}')
        
        self.pad = 0

    def patchify(self, img):
        """
            Patchify the flatmap into 48x48 patches
            If using averaging, the following patch will be shifted by an overlap_size
        """
        if self.use_avg_patch:
            cropped_patch_size = self.patch_size - self.overlap_size
            new_img_size = img.shape[1] - self.patch_size
            self.pad = cropped_patch_size - (new_img_size % cropped_patch_size)
        else:
            self.pad = self.patch_size - (img.shape[1] % self.patch_size)
        
        
        img = np.pad(img,( (0,0), (0, self.pad), (0,0)), 'symmetric')
        
        patches = []
        patch_start = 0
        while (patch_start < img.shape[1]):
            if self.use_avg_patch and patch_start > 0:
                patch_start -= self.overlap_size

            patch = self.get_patch(img, patch_start, self.patch_size)
            patches.append(patch)

            patch_start += self.patch_size
        return patches
        
    def unpatchify(self, imgs):
        """
            Stitch back the patches into flatmap/sheets
            If use averaging, the overlapping are will be averaged to avoid border artefact
        """
        if self.use_avg_patch:
            # stitching by averaging
            img = []
            buffer = []
            # ic(imgs.shape)
            for i in range(0, imgs.shape[0]):
                if i == 0:
                    patch =  imgs[i][..., :-self.overlap_size ,:]
                    end_buffer = imgs[i][..., -self.overlap_size:  ,:]

                    img = patch
                else:
                    start_buffer = imgs[i][...,:self.overlap_size,:]
                    buffer = (start_buffer + end_buffer) / 2
                    
                    patch        = imgs[i][..., self.overlap_size:-self.overlap_size ,:]
                    
                    img = np.concatenate([img, buffer, patch], axis=1)
                    # get new end buffer
                    end_buffer   = imgs[i][..., -self.overlap_size:  ,:]
                

            # in the end add end_buffer without averaging
            img = np.concatenate([img, end_buffer], axis=1)
        else:
            img = np.concatenate(imgs, axis=1)
            
        img = img[:, : -self.pad or None]
        return img

    
    def get_patch(self, img, patch_start, patch_size):
        # !! WARNING: if we use overlapping patch, make sure to use .copy()
        #  otherwise, we screw up the normalisation on the ref_coord
        return img[:, patch_start: patch_start + patch_size, :].copy()

  
    def load_patches_from_index_file(self, idx):
        """
            Load and patchify the coordinates and velocity sheets
            Note: WSS Flatmap is not patchified
        """
        xyz0, xyz1, xyz2, v1, v2, wss  = self.load_flatmap_img(self.data_filename, idx)

        # rescale the data so it fits with velocity range
        xyz0 /= self.scale_dist
        xyz1 /= self.scale_dist
        xyz2 /= self.scale_dist

        # extract patches 
        xyz0_patches = self.patchify(xyz0)
        xyz1_patches = self.patchify(xyz1)
        xyz2_patches = self.patchify(xyz2)

        v1_patches = self.patchify(v1)
        v2_patches = self.patchify(v2)
        
        for i in range(0, len(xyz0_patches)):
            # Translate ref point
            # pick reference point (x,y) = 0,0
            pos_x = 0
            pos_y = 0
            
            # get the reference point 0 on the wall
            # make sure to actually keep the value using .copy()
            ref_coord = xyz0_patches[i][pos_x,pos_y,:].copy()
            
            # recenter the coordinates to the ref point
            xyz0_patches[i] -= ref_coord
            xyz1_patches[i] -= ref_coord
            xyz2_patches[i] -= ref_coord
            
        # prepare dynamic output list
        output_list = [xyz0_patches, xyz1_patches, xyz2_patches, v1_patches, v2_patches, wss, idx]

        data_dict = {
            "xyz0": output_list[0], 
            "xyz1": output_list[1], "xyz2": output_list[2],
            "v1": output_list[3], "v2": output_list[4],
            "wss": output_list[5],
            "idx": output_list[6]
            }
        return data_dict
    
    def load_flatmap_img(self, hd5path, row_idx):
        '''
            Load the row
        '''
        idx1 = randrange(0,len(self.distances) - 1 )
        idx2 = randrange(idx1+1,len(self.distances))

        # print('load_img', hd5path, row_idx, idx1, idx2)

        with h5py.File(hd5path, 'r') as hl:
            # TODO: change to row_idx later if necessary, right now 1 file for 1 wall coordinates
            xyz0 = hl.get(self.wall_coord)[0] 

            xyz1 = hl.get(self.coord_colnames[idx1])[0]
            xyz2 = hl.get(self.coord_colnames[idx2])[0]

            v1 = hl.get(self.velocity_colnames[idx1])[row_idx]
            v2 = hl.get(self.velocity_colnames[idx2])[row_idx]
            
            wss = hl.get(self.wss_colname)[row_idx]
            
        return xyz0.astype('float32'), \
                xyz1.astype('float32'), xyz2.astype('float32'), \
                v1.astype('float32'), v2.astype('float32'), \
                wss.astype('float32')

if __name__ == '__main__':
    import config
    import matplotlib.pyplot as plt
    
    img = np.arange(0, 48*93)
    img = np.reshape(img, [48, 93])

    case_nr = '01'
    data_dir = config.DATA_DIR
    dataset_file    = f'{data_dir}/train/ch{case_nr}_clean.h5'
    
    
    scale_dist = 100
    distances = [1.0, 2.0]
    use_avg_patch = True

    pc = PatchInputHandler(dataset_file, scale_dist, 48, distances, use_avg_patch)
    data_pairs = pc.load_patches_from_index_file(0)
    

    # Load the patches of velocity sheets 
    patches = data_pairs['v1']
    
    for i in range(len(patches)):
        plt.imshow(patches[i][...,1], cmap='jet')
        plt.colorbar()
        plt.title(f'Patch {i}')
        plt.show()
    
    patches = np.array(patches)
    img = pc.unpatchify(patches)
    plt.imshow(img[...,1], cmap='jet')
    plt.colorbar()
    plt.title(f'Patch combined')
    plt.show()

    # Test the wss full flatmap
    wss = data_pairs['wss']
    plt.imshow(wss[...,1], cmap='jet')
    plt.colorbar()
    plt.title(f'WSS full flatmap')
    plt.show()
    