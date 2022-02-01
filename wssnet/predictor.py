import tensorflow as tf
import numpy as np
import os
from wssnet.Network.WSSNet import WSSNet
from wssnet.Network.PatchInputHandler import PatchInputHandler
from wssnet.utility import h5util
import h5py
import config

def prepare_network(input_shape):
    # Prepare Input 
    xyz0 = tf.keras.layers.Input(shape=input_shape + (3,), name='xyz0')
    xyz1 = tf.keras.layers.Input(shape=input_shape + (3,), name='xyz1')
    xyz2 = tf.keras.layers.Input(shape=input_shape + (3,), name='xyz2')

    v1 = tf.keras.layers.Input(shape=input_shape + (3,), name='v1')
    v2 = tf.keras.layers.Input(shape=input_shape + (3,), name='v2')

    # network & output
    input_layer = [xyz0, xyz1, xyz2, v1, v2]
    net = WSSNet()

    prediction = net.build_network(input_layer)
    model = tf.keras.Model(input_layer, prediction)

    return model

def predict_all_rows(dataset_file, scale_dist, distances):
    """
        Run predictions per row from an HDF5 file

    """
    # prepare input
    with h5py.File(dataset_file, mode = 'r' ) as hdf5:
        len_indexes = len(hdf5['wss_vector'])
        wall_coords = hdf5.get('xyz0')[0] 

    pc = PatchInputHandler(dataset_file, scale_dist, 48, distances, False)

    for i in range(len_indexes):
        print(f'Processing row {i}/{len_indexes}')
        data_pairs = pc.load_patches_from_index_file(i)
        input_data = [data_pairs['xyz0'],data_pairs['xyz1'], data_pairs['xyz2'], data_pairs['v1'], data_pairs['v2']]
        wss_true = data_pairs['wss']

        wss_pred = network.predict(input_data)
        wss_pred = pc.unpatchify(wss_pred)
        
        # save each row
        h5util.save_predictions(output_dir, output_filename, f"wss", wss_pred, compression='gzip', auto_expand=True)
        h5util.save_predictions(output_dir, output_filename, f"wss_true", wss_true, compression='gzip', auto_expand=True)
    
    # only save the wall coordinates once
    h5util.save_predictions(output_dir, output_filename, f"xyz0", wall_coords, compression='gzip', auto_expand=True)


if __name__ == '__main__':
    """
        Exampled script for prediction
    """
    # Put all your HDF5 input files here
    input_dir  = f"{config.DATA_DIR}/test" 
    # Predictions will be saved here
    output_dir = f"{config.ROOT_DIR}/results"
    
    # Put your model .h5 here
    model_path = f'{config.MODEL_DIR}/wssnet/wssnet.h5'


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
   
    # Presets, modified if necessary
    input_shape = (48,48)
    scale_dist = 100
    distances = [1.0, 2.0]

    # Load the network
    print("Loading WSSNet")
    
    network = prepare_network(input_shape)
    network.load_weights(model_path)

    # List all the case numbers
    cases = np.arange(1, 80)
    
    for i in cases:
        case_nr = f'{i:02}'

        dataset_file = f"{input_dir}/ch{case_nr}_clean.h5"
        output_filename = f'ch{case_nr}_prediction.h5'

        if not os.path.exists(dataset_file):
            # print(f"Input file does not exists: {dataset_file}")
            continue

        print(f'Processing case {case_nr}')
        print(distances)
        
        # read all the rows in the input file, and save it to output_dir/output_filename
        predict_all_rows(dataset_file, scale_dist, distances)
