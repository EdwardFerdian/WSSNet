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


# A. Training setup 
1. Download the dataset from the [figshare website](https://auckland.figshare.com/articles/software/WSSNet_aortic_4D_Flow_MRI_wall_shear_stress_estimation_neural_network/19105067).
2. Unpack it under "data" directory. 
3. The "data" directory should contains 3 csv files, and 3 subfolders (train/val/test).
4. Run the trainer.py
>> Trainer will read the CSV files and load the file and row indexes based on the ones listed in CSV files. The setup has been predefined in the code.

# B. Prediction setup 
1.  Download the pre-trained weights from the [figshare website](https://auckland.figshare.com/articles/software/WSSNet_aortic_4D_Flow_MRI_wall_shear_stress_estimation_neural_network/19105067).
2. Unpack it under "models" directory.
3. Run predictor.py
>> This will run a prediction of an example input file provided in the examples/ directory. Please refer to **section E** below for full description.

# C. Prepare data
We are using aortic surface template based on [Liang et al.'s work](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5630492/#R30). The original template was obtained from [here](https://github.com/TML-Gatech/DL_Stress_TAA) where it consists of 50x100 grid.

We further modified the template (into 48x93) and performed UV unwrapping, which can be find in the templates directory.
The template is used to register to patient-spesific geometry from segmentations.
Registration was performed using [PyCPD](https://github.com/siavashk/pycpd). A modified version of the code is already included in wssnet/pycpd directory.
# D. Prepare your own training dataset
This is an example on how to prepare the WSS and velocity 'sheets' from CSV files (exported from CFD simulations). Example csv files can be downloaded from the [figshare website](https://auckland.figshare.com/articles/software/WSSNet_aortic_4D_Flow_MRI_wall_shear_stress_estimation_neural_network/19105067).

Note:
1. Each time frame in the simulation is exported as a single csv file. We split the csv file into 2 parts:
    - **export-{time_index}.csv**: contains the mesh coordinates (x,y,z) and velocity at the corresponding coordinates (vx, vy, vz, |v|)
    - **export_wall-{time_index}.csv**: contains the wall coordinates (x, y, z) and wall shear stress at the corresponding coordinates (wssx, wssy, wssz, |wss|)

Prepare velocity sheets directly from csv files:
- Directly run the prepare_csv2sheet.py script using the predefined value

Prepare velocity sheets from an interpolated grid:
- Run the prepare_csv2grid.py script using the predefined value
- Run the prepare_mri_sheet.py script (see **Section E-3**)

Extracting this data into sheets also require the geometry. An example geometry is provided. See **Section E-2.5**

# E. Running prediction on MRI data
Some of the preparation steps, such as segmentation are still manual. Please refer to the following steps.
For some of the steps below, example files are provided in the examples/ directory.
## 1. Sort and build the mri data
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
* For WSSNet, only Phase images are needed. This script is the same as the one in 4DFlowNet, where magnitude images are included.

## 2. Prepare aorta geometry
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

## 3. Prepare coordinates and velocity sheets
This step requires two files:
* A 3D velocity image (see step 1). Example provided: **example_grid.h5**
* Surface mesh (see step 2). Example provided: **example_aorta_reg.h5**

To extract the coordinates and velocity sheets, run the *prepare_mri_sheet.py*.    
Please fill in the input parameters accordingly. Predefined values have been set using the example files.
<br/>
Example output file provided: **example_sheet.h5**


# F. Visualization
Under visualize, we provided 3 example scripts:
- *plot_tawss_osi_flatmap.py* to plot TAWSS and OSI as 2D flatmaps
- *plot_tawss_osi_3d.py* to plot TAWSS and OSI in 3D point clouds
<p align="middle">
    <img src="https://i.imgur.com/ciijrxh.png" width="300">
</p>
- *plot_wss_3d.py* to plot WSS with a slider through time frames
<p align="middle">
    <img src="https://i.imgur.com/cmTWCnm.png" width="300">
</p>


An example prediction result file is provided in: **examples/case70_prediction.h5**

TODO: Plot as 3D surface instead of point clouds

## Contact Information

If you encounter any problems in using the code, please open an issue in this repository or feel free to contact me by email.

Author: Edward Ferdian

Mail: e.ferdian@auckland.ac.nz
| edwardferdian03@gmail.com
