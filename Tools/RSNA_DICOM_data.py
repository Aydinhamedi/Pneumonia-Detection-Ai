import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Directory containing the DICOM files
dcm_dir = 'stage_2_train_images'

# Read the CSV file
df = pd.read_csv('stage_2_detailed_class_info.csv')

# Directories for the three classes
not_normal_dir = 'database/Not Normal'
normal_dir = 'database/NORMAL'
pneumonia_dir = 'database/PNEUMONIA'

# Create the directories if they don't exist
os.makedirs(not_normal_dir, exist_ok=True)
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(pneumonia_dir, exist_ok=True)

# Get the list of files
files = [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')]

# Initialize the progress bar
pbar = tqdm(total=len(files), desc='Processing DICOM files')

# Loop over all files in the directory
for filename in files:
    # Read the DICOM file
    dcm = pydicom.dcmread(os.path.join(dcm_dir, filename))
    # Get the pixel array from the DICOM file
    pixels = dcm.pixel_array
    # Convert the pixel array to an image
    img = Image.fromarray(pixels)
    # Get the label for this file
    label = df[df['patientId'] == filename[:-4]]['class'].values[0]
    # Save the image to the appropriate directory with the original filename
    if label == 'No Lung Opacity / Not Normal':
        img.save(os.path.join(not_normal_dir, filename[:-4] + '.jpeg'))
        # print(f'Saved {filename[:-4]}.jpeg to {not_normal_dir}')
    elif label == 'Normal':
        img.save(os.path.join(normal_dir, filename[:-4] + '.jpeg'))
        # print(f'Saved {filename[:-4]}.jpeg to {normal_dir}')
    else:  # 'Lung Opacity'
        img.save(os.path.join(pneumonia_dir, filename[:-4] + '.jpeg'))
        # print(f'Saved {filename[:-4]}.jpeg to {pneumonia_dir}')
    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()
