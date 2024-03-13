import os
import glob
import shutil

# Define the backup directory and main directory paths
backup_dir = 'backup'
main_dir = ''

# List all folders in the backup directory
folders = [f for f in os.listdir(backup_dir) if os.path.isdir(os.path.join(backup_dir, f))]

# Display the folders with their IDs
print('Backup List:')
for i, folder in enumerate(folders, start=1):
    print(f' -- [{i}] > `{folder}`')

# Prompt the user to input an ID
user_input = int(input('Enter the ID of the folder you want to copy from: '))

# Check if the input is valid
if 1 <= user_input <= len(folders):
    # Get the selected folder
    selected_folder = folders[user_input - 1]
    
    # Construct the source and destination paths
    source_path = os.path.join(backup_dir, selected_folder, '*.ipynb')
    destination_path = os.path.join(main_dir, f'Backup_{selected_folder}_COPY.ipynb')
    
    # Copy the .ipynb file to the main directory with the new name
    for file in glob.glob(source_path):
        shutil.copy(file, destination_path)
        print(f'Copied {file} to {destination_path}')
else:
    print('Invalid ID. Please try again.')
