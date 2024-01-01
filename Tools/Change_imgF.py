import os
import uuid
import shutil
from PIL import Image
from tqdm import tqdm

def convert_image_format(image_path, output_path, output_format):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert the image to the desired format and save it
        img.save(output_path, output_format)

def convert_images_in_dir(input_dir, output_dir, output_format):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all files in the input directory
    filenames = os.listdir(input_dir)

    # Create a progress bar
    with tqdm(total=len(filenames), desc="Converting images", ncols=75, unit='img') as pbar:
        # Iterate over all files in the input directory
        for filename in filenames:
            # Create the full input path
            input_path = os.path.join(input_dir, filename)
            
            # Generate a random unique name for the output file
            unique_name = str(uuid.uuid4())
            output_path = os.path.join(output_dir, f"{unique_name}.{output_format}")

            # Check if the image is already in the desired format
            if filename.lower().endswith(f".{output_format}"):
                # If it is, copy the image to the new location
                # Use the original file extension for the output path
                original_extension = os.path.splitext(filename)[1]
                shutil.copy(input_path, os.path.join(output_dir, f"{unique_name}{original_extension}"))
            else:
                # If it's not, open and save the image in the new format
                with Image.open(input_path) as img:
                    img.save(output_path, output_format)

            # Update the progress bar
            pbar.update(1)

print('NORMAL conv...')
convert_images_in_dir('Database\\Train\\Data\\train\\NORMAL', 'Database\\Train\\Data\\train\\NORMAL_NEW', 'jpeg')
print('PNEUMONIA conv...')
convert_images_in_dir('Database\\Train\\Data\\train\\PNEUMONIA_NEW', 'Database\\Train\\Data\\train\\PNEUMONIA_NEW', 'jpeg')
print('done.')
