import numpy as np
from wssnet.Network.TrainerController import TrainerController
from wssnet.Network.CsvInputHandler import CsvInputHandler
import config

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":
    data_dir = config.DATA_DIR

    train_dir = f'{data_dir}/train'
    val_dir = f'{data_dir}/train'
    test_dir = f'{data_dir}/train'

    restore = False
    if restore:
        model_dir  = "[model_dir]"
        model_file = "[model_name]-best.h5"
    
    # csv index file
    training_file = f'{config.DATA_DIR}/testval.csv'
    validate_file = f'{config.DATA_DIR}/testval.csv'
    test_file = f'{config.DATA_DIR}/testval.csv'
    
    
    QUICKSAVE = True
    lr_decay = 'cosine'
    
    # Hyperparameters optimisation variables
    initial_learning_rate = 1e-4
    epochs =  2
    batch_size = 16

    # Network setting
    network_name = 'test'
    input_shape = (48,48)
    
    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = CsvInputHandler(train_dir, True, batch_size)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=None)

    # VALIDATION iterator
    valdh = CsvInputHandler(val_dir, False, batch_size)
    valset = valdh.initialize_dataset(valset, shuffle=True, n_parallel=None)

    # # Bechmarking dataset, use to keep track of prediction progress per best model
    testset = None
    if QUICKSAVE and test_file is not None:
        testset = load_indexes(test_file)
        # WE use this bechmarking set so we can see the prediction progressing over time
        ph = CsvInputHandler(test_dir, False, batch_size)
        # No shuffling, so we can save the first batch consistently
        testset = ph.initialize_dataset(testset, shuffle=False) 


    
    # ------- Main Network ------
    print(f"Flat WSSNet {input_shape}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController(input_shape, initial_learning_rate, lr_decay, QUICKSAVE, network_name)
    network.init_model_dir()

    if restore:
        print("Restoring model...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())

    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)
