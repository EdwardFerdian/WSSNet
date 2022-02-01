import tensorflow as tf
import numpy as np
import h5py
from random import randrange, randint
import cv2
from utility import augmentation as aug

class CsvInputHandler():
    # constructor
    def __init__(self, data_dir, use_augment, batch_size):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.scale_dist = 100

        self.use_augment = use_augment
        
        self.wall_coord  = 'xyz0'
        self.wss_colname = 'wss_vector'
        self.mask_colname = 'wss_mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        # u, v, mag, venc, dx
        output_datatypes = [tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32, tf.float32, tf.int32]
        
        flattened_output = tf.py_function(func=self.load_patches_from_index_file, 
            inp=[indexes], Tout=output_datatypes)

        data_dict = {
            "xyz0": flattened_output[0], 
            "xyz1": flattened_output[1], "xyz2": flattened_output[2],
            "v1": flattened_output[3], "v2": flattened_output[4],
            "wss": flattened_output[5], "mask": flattened_output[6],
            "idx": flattened_output[7]
            }
       
        return data_dict
    
    def augment_roll(self, shift, xyz0, xyz1, xyz2, v1, v2, wss, mask):
        # shift roll
        xyz0 = np.roll(xyz0, shift, axis=0)
        xyz1 = np.roll(xyz1, shift, axis=0)
        xyz2 = np.roll(xyz2, shift, axis=0)

        v1 = np.roll(v1, shift, axis=0)
        v2 = np.roll(v2, shift, axis=0)

        wss = np.roll(wss, shift, axis=0)
        mask = np.roll(mask, shift, axis=0)
        return xyz0, xyz1, xyz2, v1, v2, wss, mask

    def augment_noise(self, v1, v2):
        rnd = randint(0,1)  
        # 50% chance to add noise
        if rnd > 0:
            # noise between 1-4% venc
            noiseLevel = randint(1,4)
            maxStd = 1.5 * noiseLevel / 100. 
            
            # print('ADDING noise', maxStd)
            # we assume the same noise on every sheet
            # although differnt on each component
            noise = np.random.normal(0, maxStd, v1.shape)
            for c in range(0,3):
                noise[...,c] = cv2.GaussianBlur(noise[...,c], (3,3), 0)

            v1 += noise
            v2 += noise

        return v1, v2


    def augment_rotation(self, xyz0, xyz1, xyz2, v1, v2, wss):
        rnd = randint(0,4)  
        if rnd == 0:
            # print('no rotation')
            return xyz0, xyz1, xyz2, v1, v2, wss
        else:
            randAngle = randrange(0, 360)
            randAxis  = randint(0,2)
            # perform augmentation 
            # TODO: separate rotation matrix
            xyz0 = aug.rotate(xyz0, randAngle, axis=randAxis)
            xyz1 = aug.rotate(xyz1, randAngle, axis=randAxis)
            xyz2 = aug.rotate(xyz2, randAngle, axis=randAxis)
            v1 = aug.rotate(v1, randAngle, axis=randAxis)
            v2 = aug.rotate(v2, randAngle, axis=randAxis)
            wss = aug.rotate(wss, randAngle, axis=randAxis)

        return xyz0, xyz1, xyz2, v1, v2, wss
        
    def get_patch(self, img, patch_start, patch_size):
        return img[:, patch_start: patch_start + patch_size]

    def get_patches(self, xyz0, xyz1, xyz2, v1, v2, wss, mask, randomize):
        # prepare random start index for patch
        patch_size = xyz0.shape[0]
        if randomize:
            # start from col 3 instead of 0 because of high WSS
            patch_start = randrange(3, xyz0.shape[1]-patch_size) # (shape0 x shape0)
        else:
            patch_start = 3
        
        # get patch
        xyz0 = self.get_patch(xyz0, patch_start, patch_size)
        xyz1 = self.get_patch(xyz1, patch_start, patch_size)
        xyz2 = self.get_patch(xyz2, patch_start, patch_size)
        v1 = self.get_patch(v1, patch_start, patch_size)
        v2 = self.get_patch(v2, patch_start, patch_size)
        wss = self.get_patch(wss, patch_start, patch_size)
        mask = self.get_patch(mask, patch_start, patch_size)

        return xyz0, xyz1, xyz2, v1, v2, wss, mask

    def load_patches_from_index_file(self, csv_row):
        # get the content of csv row
        h5filename = '{}/{}'.format(self.data_dir, bytes.decode(csv_row[0].numpy()))
        idx = int(csv_row[1])
        dist1 = float(csv_row[2])
        dist2 = float(csv_row[3])

        xyz0, xyz1, xyz2, v1, v2, wss, mask  = self.load_sheet_data(h5filename, idx, dist1, dist2)

        # extract patches 
        xyz0, xyz1, xyz2, v1, v2, wss, mask = self.get_patches(xyz0, xyz1, xyz2, v1, v2, wss, mask, self.use_augment)

        if self.use_augment:
            # Augmentation - Rotate
            xyz0, xyz1, xyz2, v1, v2, wss = self.augment_rotation(xyz0, xyz1, xyz2, v1, v2, wss)

            # Augmentation - Noise
            v1, v2 = self.augment_noise(v1, v2)

        # Augmentation - Roll
        shift = randint(-5, 5)
        xyz0, xyz1, xyz2, v1, v2, wss, mask = self.augment_roll(shift, xyz0, xyz1, xyz2, v1, v2, wss, mask)


        # Augmentation - Translate ref point
        # pick a random point in xyz0
        pos_x = randrange(0, xyz0.shape[0])
        pos_y = randrange(0, xyz0.shape[1])
        
        # get the reference point 0 on the wall
        # make sure to actually keep the value using .copy()
        ref_coord = xyz0[pos_x,pos_y,:].copy()
        
        # recenter the coordinates to the ref point
        xyz0 -= ref_coord
        xyz1 -= ref_coord
        xyz2 -= ref_coord

        # rescale the data so it fits with velocity range
        xyz0 /= self.scale_dist
        xyz1 /= self.scale_dist
        xyz2 /= self.scale_dist

        # prepare dynamic output list
        output_list = [xyz0, xyz1, xyz2, v1, v2, wss, mask[..., tf.newaxis], idx]

        return output_list
    
    def load_sheet_data(self, hd5path, row_idx, dist1, dist2):
        '''
            
        '''
        # print('load_sheet_data', hd5path, row_idx, dist1, dist2)

        with h5py.File(hd5path, 'r') as hl:
            # TODO: change to row_idx later if necessary, right now 1 file for 1 wall coordinates
            xyz0 = hl.get(self.wall_coord)[0] 

            xyz1 = hl.get(f'xyz{dist1}')[0]
            xyz2 = hl.get(f'xyz{dist2}')[0]

            v1 = hl.get(f'v{dist1}')[row_idx]
            v2 = hl.get(f'v{dist2}')[row_idx]
            
            wss = hl.get(self.wss_colname)[row_idx]
            mask = hl.get(self.mask_colname)[0]
            
        return xyz0.astype('float32'), \
                xyz1.astype('float32'), xyz2.astype('float32'), \
                v1.astype('float32'), v2.astype('float32'), \
                wss.astype('float32'), mask.astype('float32')
