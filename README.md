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
3. Run >> pip install -r requirements.txt
4. Optional: open config.py to setup the directories. They have been predefined for you. The manual below is set based on these predefined directories.


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

# Prepare data
We are using aortic surface template based on [Liang et al.'s work](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5630492/#R30). The original template was obtained from [here](https://github.com/TML-Gatech/DL_Stress_TAA) where it consists of 50x100 grid.

We further modified the template (into 48x93) and performed UV unwrapping, which can be find in the templates directory.
The template is used to register to patient-spesific geometry from segmentations.
Registration was performed using [PyCPD](https://github.com/siavashk/pycpd). A modified version of the original code is already included in wssnet/pycpd directory.
# Prepare your own training dataset
TBA



# Running prediction on MRI data
Some of the preparation steps are still manual. Please refer to the following steps.
## Sort and build the mri data
To prepare 4D Flow MRI data to HDF5, go to the prepare_data/ directory and run the following script:

    >> python prepare_data.py --input-dir [4DFlowMRI_CASE_DIRECTORY]

    >> usage: prepare_mri_data.py [-h] --input-dir INPUT_DIR
                           [--output-dir OUTPUT_DIR]
                           [--output-filename OUTPUT_FILENAME]
                           [--phase-pattern PHASE_PATTERN]
                           [--mag-pattern MAG_PATTERN] [--fh-mul FH_MUL]
                           [--rl-mul RL_MUL] [--in-mul IN_MUL]

Notes: 
*  The directory must contains the following structure:
    [CASE_NAME]/[Magnitude_or_Phase]/[TriggerTime]
* There must be exactly 3 Phase and 3 Magnitude directories. If you have only 1 magnitude directory, you need to adjust the code.
* To get the required directory structure, [DicomSort](https://dicomsort.com/) is recommended. Sort by SeriesDescription -> TriggerTime.
* In our case, VENC and velocity direction is read from the SequenceName DICOM HEADER. Code might need to be adjusted if the criteria is different.

## Prepare aorta geometry
For this step, some examples are already provided in the "examples" directory.
1. Build a temporal max or mean PC-MRA (script not included)
2. Segment the aorta without any branch and export as STL.    
Example provided: **example_segmentation.stl**
3. In Blender, truncate ascending and descending aorta.    
Example provided: **example_template.blend**
4. Export as OBJ file, ensure *Apply Modifiers, Include UVs, and Keep Vertex Order* are selected.   
Example provided **example_aorta.obj**
5. Register the template mesh to the truncated mesh by running the *prepare_mesh/register_mesh.py*   
The example of registered mesh provided as **example_aorta_reg.obj**



## Prepare data from MRI 

TBA

## Prediction

TBA


## Contact Information

If you encounter any problems in using the code, please open an issue in this repository or feel free to contact me by email.

Author: Edward Ferdian (edwardferdian03@gmail.com).
