# WSSNet

Spatiotemporal Aortic Wall Shear Stress estimation from 4D Flow MRI
--



This is an implementation of the paper [WSSNet: Aortic wall shear stress estimation using deep learning on 4D Flow MRI](https://www.frontiersin.org/articles/10.3389/fcvm.2021.769927/full) using Tensorflow 2.2.0. 


WSSNet predicts WSS vectors per time frame from 4D Flow MRI dataset. It allows accurate prediction of WSS from native MRI resolution.

<p align="left">
    <img src="https://i.imgur.com/f7IdEhf.png" width="600">
</p>

The pre-trained network weights and training dataset are available
[here](https://auckland.figshare.com/articles/software/WSSNet_aortic_4D_Flow_MRI_wall_shear_stress_estimation_neural_network/19105067).

# Package installation

Install the wssnet as a package. 
1. GO to the directory where the setup.py is located
2. Run >> pip install -e .
3. Optional: open config.py to setup the directories. They have been predefined for you. The manual below is set based on these predefined directories.


# Training setup 
1. Download the dataset from the [figshare website](https://auckland.figshare.com/articles/software/WSSNet_aortic_4D_Flow_MRI_wall_shear_stress_estimation_neural_network/19105067).
2. Unpack it under "data" directory. 
3. The "data" directory should contains 3 csv files, and 3 subfolders (train/val/test).
4. Run the trainer.py
>> Trainer will read the CSV files and load the file and row indexes based on the ones listed in CSV files. The setup has been predefined in the code.

# Prediction setup 
1.  Download the pre-trained weights from the [figshare website](https://auckland.figshare.com/articles/software/WSSNet_aortic_4D_Flow_MRI_wall_shear_stress_estimation_neural_network/19105067).
2. Unpack it under "models" directory.
3. Run predictor.py
>> Assuming the dataset have been downloaded, the prediction script will predict everything under the "data/test" directory.

# Prepare your own training dataset
TBA



# Running prediction on MRI data
## Prepare data from MRI 

TBA

## Prediction

TBA


## Contact Information

If you encounter any problems in using the code, please open an issue in this repository or feel free to contact me by email.

Author: Edward Ferdian (edwardferdian03@gmail.com).
