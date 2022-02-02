
import numpy as np
import pydicom
import os
import re
from wssnet.utility import h5util
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Input directory contains the Dicom files with [Phase/Magnitude series desc]/[triggerTime] structure')
    parser.add_argument('--output-dir', type=str, default='data_mri', help='Output directory where the dataset will be saved')
    parser.add_argument('--output-filename', type=str, default='data_mri_grid.h5', help='Output filename (HDF5)')
    parser.add_argument('--phase-pattern', type=str, default='_P_', help='Pattern of phase series description')
    parser.add_argument('--mag-pattern', type=str, default='_M_', help='Pattern of magnitude series description')
    parser.add_argument('--fh-mul', type=int, default= -1, help='Velocity multiplier for Foot-Head direction')
    parser.add_argument('--rl-mul', type=int, default=  1, help='Velocity multiplier for Right-Left direction')
    parser.add_argument('--in-mul', type=int, default=  1, help='Velocity multiplier for Inplane direction')
    args = parser.parse_args()

    # Pattern to ignore
    pattern_ignore = 'phantom'
    

    case_dir     = args.input_dir
    phase_pattern = args.phase_pattern
    mag_pattern   = args.mag_pattern
    output_path   = args.output_dir
    output_filename = args.output_filename

    in_multiplier = args.in_mul
    fh_multiplier = args.fh_mul
    rl_multiplier = args.rl_mul
    
    output_filepath = f'{output_path}/{output_filename}'

    # 1. Get the phase and magnitude directories 
    directories = os.listdir(case_dir)
    phase_dirs = [item for item in directories if phase_pattern in item]
    mag_dirs   = [item for item in directories if mag_pattern   in item]

    # Filter out pattern (upper case only)
    phase_dirs = [item for item in phase_dirs if not pattern_ignore.upper() in item.upper()]
    mag_dirs   = [item for item in mag_dirs if not pattern_ignore.upper() in item.upper()]

    print(f"Processing case {case_nr}")
    print("Phase dirs:\n", "\n".join(phase_dirs))
    print("Mag dirs:\n"  , "\n".join(mag_dirs))

    assert len(phase_dirs) == 3, f"There must be exactly 3 Phase directories with {phase_pattern} pattern"
    assert len(mag_dirs)   == 3, f"There must be exactly 3 Magnitude directories with {mag_pattern} pattern"

    # 2. Get and sort the trigger times
    dirpath = f'{case_dir}/{phase_dirs[0]}'
    timeFrames = [d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))]
    # print(timeFrames)
    timeFrames = sorted(timeFrames, key=float)
    print('All frames sorted:', timeFrames)

    # Create the output dir
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 3. Looping through the time frames
    for j in range(0, len(timeFrames)):
        triggerTime = timeFrames[j]
        print(f"\rProcessing {j+1}/{len(timeFrames)} (frame {triggerTime})          ", end="\r")

        # Wrap it as DicomData instance
        dicom_data = DicomData()
        # Collect the 3 phase and magnitude volumes for 1 time frame
        for mag_dir, p_dir in zip(mag_dirs, phase_dirs):
            magnitude_path = f'{case_dir}/{mag_dir}/{triggerTime}'
            phase_path = f'{case_dir}/{p_dir}/{triggerTime}'

            # Get the magnitude and phase images
            mag_images, _, _, _                = get_volume(magnitude_path)
            phase_images, spacing, sequence, origin = get_volume(phase_path)
            
            dicom_data._phaseImages.append(phase_images)
            dicom_data._magImages.append(mag_images)
            dicom_data.sequenceNames.append(sequence)
            dicom_data.spacing = spacing
            dicom_data.origin  = origin
            # print(origin)
        # Save per row
        dicom_data.determine_velocity_components(in_multiplier, fh_multiplier, rl_multiplier)
        dicom_data.save_dataset(output_filepath, triggerTime)

        # break
    # End of trigger time loop
    print(f'\nDone! saved at {output_filepath}')


def get_filepaths(directory):
    """
        This function will generate the file names in a directory 
        tree by walking the tree either top-down or bottom-up. For each 
        directory in the tree rooted at directory top (including top itself), 
        it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths 

def get_volume(vol_dir):
    """
        Get phase or magnitude image volume
    """
    volume = []
    # Retrieve all the dicom filepaths
    files  = get_filepaths(vol_dir)
    
    for slice_nr, dicom_path in enumerate(files):
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        
        if slice_nr == 0:
            # Get this on the first slice only
            spacing = ds.PixelSpacing
            spacing.insert(0, ds.SliceThickness)
            spacing = np.asarray(spacing)

            origin = ds.ImagePositionPatient
            
            # Note: In our case, sequence name contains venc and direction info
            sequence_name = ds.SequenceName
            # print(sequence_name)

        volume.append(img)
    volume = np.asarray(volume)
    return volume, spacing, sequence_name, origin


class DicomData:
    def __init__(self):
        self.sequenceNames = []
        self.spacing = []
        self.origin = []

        self._phaseImages = []
        self._magImages = []

        # vel and mag Components
        self.u = None
        self.v = None
        self.w = None

        self.u_mag = None
        self.v_mag = None
        self.w_mag = None

        self.u_venc = None
        self.v_venc = None
        self.w_venc = None
    
    def print(self):
        attributes = vars(self)
        for item in attributes:
            if not item.startswith('_'):
                print (item , ' : ' , attributes[item])

    def phase_to_velocity(self, phase_image, venc):
        """
            Phase image range: 0-4096, with 2048 as 0 velocity (in m/s)
        """
        return (phase_image - 2048.) / 2048. * venc / 100.

    def determine_velocity_components(self, in_multiplier, fh_multiplier, rl_multiplier):
        """ 
            Determine the velocity direction and venc from the sequence name 
        """
        # print("Calculating velocity components...")
        for i in range(len(self._phaseImages)):
            seqName = self.sequenceNames[i]
            phase_image = self._phaseImages[i]
            mag_image = self._magImages[i]

            # Check venc from the sequence name (e.g. fl3d1_v150fh)
            pattern = re.compile(".*?_v(\\d+)(\\w+)")
            found = pattern.search(seqName)
            
            assert found, "Venc pattern not found, please check your DICOM header."
        
            venc = int(found.group(1))
            direction = found.group(2)
            # print('venc_direction', direction, venc)

            # Convert the phase image to velocity
            phase_image = self.phase_to_velocity(phase_image, venc)

            # Note: This is based on our DICOM header. The direction is a bit confusing.
            # In our case we always have in/ap/fh combination
            if direction == "in":
                self.u      = phase_image * in_multiplier
                self.u_mag  = mag_image
                self.u_venc = venc/100
            elif direction == "rl" or direction == "ap":
                self.w      = phase_image * rl_multiplier
                self.w_mag  = mag_image
                self.w_venc = venc/100
            else: # "fh" 
                self.v      = phase_image * fh_multiplier
                self.v_mag  = mag_image
                self.v_venc = venc/100

    def save_dataset(self, output_filepath, triggerTime):
        assert self.u is not None, "Please calculate velocity components first"

        h5util.save_to_h5(output_filepath, "triggerTimes", float(triggerTime), compression='gzip')
       
        h5util.save_to_h5(output_filepath, "u", self.u, compression='gzip')
        h5util.save_to_h5(output_filepath, "v", self.v, compression='gzip')
        h5util.save_to_h5(output_filepath, "w", self.w, compression='gzip')

        h5util.save_to_h5(output_filepath, "mag_u", self.u_mag, compression='gzip')
        h5util.save_to_h5(output_filepath, "mag_v", self.v_mag, compression='gzip')
        h5util.save_to_h5(output_filepath, "mag_w", self.w_mag, compression='gzip')

        h5util.save_to_h5(output_filepath, "venc_u", self.u_venc, compression='gzip')
        h5util.save_to_h5(output_filepath, "venc_v", self.v_venc, compression='gzip')
        h5util.save_to_h5(output_filepath, "venc_w", self.w_venc, compression='gzip')

        h5util.save_to_h5(output_filepath, "dx", self.spacing, compression='gzip')
        h5util.save_to_h5(output_filepath, "origin", self.origin, compression='gzip')

if __name__ == '__main__':
    main()