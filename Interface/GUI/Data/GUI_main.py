# Copyright (c) 2023 Aydin Hamedi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# start L1
print('Loading the GUI...', end='\r')
# pylib
import os
import re
import cv2
import sys
import cpuinfo
import difflib
import inspect
import traceback
import subprocess
import requests
from tqdm import tqdm
import PySimpleGUI as sg  
from loguru import logger
import efficientnet.tfkeras
from tkinter import filedialog
from datetime import datetime
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Utils
from Utils.one_cycle import OneCycleLr
from Utils.lr_find import LrFinder
from Utils.Grad_cam import make_gradcam_heatmap
from Utils.print_color_V2_NEW import print_Color_V2
from Utils.print_color_V1_OLD import print_Color
from Utils.Other import *
# global vars>>>
# CONST SYS
GUI_Ver = '0.8.9.3'
Model_dir = 'Data/PAI_model'  # without file extention
Database_dir = 'Data/dataset.npy'
IMG_AF = ('JPEG', 'PNG', 'BMP', 'TIFF', 'JPG')
Github_repo_Releases_Model_name = 'PAI_model_T.h5'
Github_repo_Releases_Model_light_name = 'PAI_model_light_T.h5'
Github_repo_Releases_URL = 'https://api.github.com/repos/Aydinhamedi/Pneumonia-Detection-Ai/releases/latest'
Model_FORMAT = 'H5_SF'  # TF_dir/H5_SF
IMG_RES = (224, 224, 3)
train_epochs_def = 4
SHOW_CSAA_OS = False
# normal global
img_array = None
label = None
model = None
# Other
logger.remove()
logger.add('Data\\logs\\SYS_LOG_{time}.log',
           backtrace=True, diagnose=True, compression='zip')
logger.info('GUI Start...\n')
tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
# crator
CSAA = '''
~*
  ___                              _        ___      _          _   _              _   _   ___      _ 
 | _ \_ _  ___ _  _ _ __  ___ _ _ (_)__ _  |   \ ___| |_ ___ __| |_(_)___ _ _     /_\ (_) | _ )_  _(_)
 |  _/ ' \/ -_) || | '  \/ _ \ ' \| / _` | | |) / -_)  _/ -_) _|  _| / _ \ ' \   / _ \| | | _ \ || |_ 
 |_| |_||_\___|\_,_|_|_|_\___/_||_|_\__,_| |___/\___|\__\___\__|\__|_\___/_||_| /_/ \_\_| |___/\_, (_)
                                                                                               |__/   
~*    _   ____ ___ 
   /_\ |__  | __|
  / _ \  / /|__ \\
 /_/ \_\/_/ |___/                                                                                                                                               
'''
# GUI logo
GUI_text_logo = '''
~*
  _______  __    __   __     .___  ___.   ______    _______   _______ 
 /  _____||  |  |  | |  |    |   \/   |  /  __  \  |       \ |   ____|
|  |  __  |  |  |  | |  |    |  \  /  | |  |  |  | |  .--.  ||  |__   
|  | |_ | |  |  |  | |  |    |  |\/|  | |  |  |  | |  |  |  ||   __|  
|  |__| | |  `--'  | |  |    |  |  |  | |  `--'  | |  '--'  ||  |____ 
 \______|  \______/  |__|    |__|  |__|  \______/  |_______/ |_______|
~*                                                                      
  ______   .__   __.                                                  
 /  __  \  |  \ |  |                                                  
|  |  |  | |   \|  |                                                  
|  |  |  | |  . `  |                                                  
|  `--'  | |  |\   |                                                  
 \______/  |__| \__|                                                  
                          
'''
# HF>>>
# open_file_GUI
def open_file_GUI():
    """Opens a file selection dialog GUI to allow the user to select an image file.

    Builds a filetypes filter from the IMG_AF global variable, joins the extensions 
    together into a filter string, converts to lowercase. Opens the file dialog, 
    and returns the selected file path if one was chosen.

    Returns:
        str: The path to the selected image file, or None if no file was chosen.
    """
    formats = ";*.".join(IMG_AF)
    formats = "*." + formats.lower()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", formats)])
    if file_path:
        return file_path

# download_file_from_github
def download_file_from_github(url: str, file_name: str, save_as: str, chunk_size: int):
    """Downloads a file from a GitHub release API URL to a local path.

    Args:
        url (str): The GitHub API URL for the release to download from.
        file_name (str): The name of the file to download from the release.
        save_as (str): The local path to save the downloaded file to.
        chunk_size (int): The chunk size to use when streaming the download.
    """
    response = requests.get(url)
    data = response.json()
    logger.debug(f'download_file_from_github:data(json) {data}')
    # Get the name of the latest release
    release_name = data['name']
    print(f'Latest release: {release_name}')

    # Get the assets of the latest release
    assets = data['assets']

    # Find the required asset in the assets
    for asset in assets:
        if asset['name'] == file_name:
            download_url = asset['browser_download_url']
            break
    if 'download_url' in locals():
        # Download the file with a progress bar
        response = requests.get(download_url, stream=True)
        file_size = int(response.headers['Content-Length'])
        progress_bar = tqdm(total=file_size, unit='b', unit_scale=True)

        with open(save_as, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                f.write(chunk)

        progress_bar.close()

        if file_size != 0 and progress_bar.n != file_size:
            print_Color('~*ERROR: ~*Something went wrong while downloading the file.', ['red', 'yellow'], advanced_mode=True)
            logger.warning('download_file_from_github>>ERROR: Something went wrong while downloading the file.')
        else:
            print(f"File '{save_as}' downloaded successfully.")
            logger.debug(f"download_file_from_github>>Debug: File '{save_as}' downloaded successfully.")
    else:
        print_Color('~*ERROR: ~*Something went wrong while finding the file.', ['red', 'yellow'], advanced_mode=True)
        logger.warning('download_file_from_github>>ERROR: Something went wrong while finding the file.')

# CF>>>
# CI_ulmd
def CI_ulmd():
    print_Color(
        'Warning: upload model data set (currently not available!!!)',
        ['yellow'])

# CI_pwai
def CI_pwai(Auto: bool = False):
    # global var import
    global model
    # check for input img
    if img_array is not None:
        try:
            if model is None:
                print_Color('loading the Ai model...', ['normal'])
                model = load_model(Model_dir)
        except (ImportError, IOError):
            print_Color('~*ERROR: ~*Failed to load the model. Try running `uaim` first.',
                        ['red', 'yellow'], advanced_mode=True)
        else:
            print_Color('predicting with the Ai model...', ['normal'])
            model_prediction_ORG = model.predict(img_array)
            model_prediction = np.argmax(model_prediction_ORG, axis=1)
            pred_class = 'PNEUMONIA' if model_prediction == 1 else 'NORMAL'
            class_color = 'red' if model_prediction == 1 else 'green'
            confidence = np.max(model_prediction_ORG)
            print_Color(f'~*the Ai model prediction: ~*{pred_class}~* with confidence ~*{confidence:.2f}~*.',
                        ['normal', class_color, 'normal', 'green', 'normal'], advanced_mode=True)
            if confidence < 0.82:
                print_Color('~*WARNING: ~*the confidence is low.',
                            ['red', 'yellow'], advanced_mode=True)
            if model_prediction == 1:
                if not Auto:
                    print_Color('~*Do you want to see a Grad cam of the model? ~*[~*Y~*/~*n~*]: ',
                            ['yellow', 'normal', 'green', 'normal', 'red', 'normal'],
                            advanced_mode=True,
                            print_END='')
                    Grad_cam_use = input('')
                else:
                    Grad_cam_use = 'y'    
                if Grad_cam_use.lower() == 'y':
                    clahe = cv2.createCLAHE(GUIpLimit=1.8)
                    Grad_cam_heatmap = make_gradcam_heatmap(img_array,
                                                            model, 'top_activation',
                                                            second_last_conv_layer_name = 'top_conv',
                                                            sensitivity_map = 2, pred_index=tf.argmax(model_prediction_ORG[0])) 
                    Grad_cam_heatmap = cv2.resize(np.GUIp(Grad_cam_heatmap, 0, 1), (img_array.shape[1], img_array.shape[2]))
                    Grad_cam_heatmap = np.uint8(255 * Grad_cam_heatmap)
                    Grad_cam_heatmap = cv2.applyColorMap(Grad_cam_heatmap, cv2.COLORMAP_VIRIDIS)
                    Grad_cam_heatmap = np.GUIp(np.uint8((Grad_cam_heatmap * 0.3) + ((img_array * 255) * 0.7)), 0, 255)
                    # Resize the heatmap for a larger display
                    display_size = (600, 600)  # Change this to your desired display size
                    Grad_cam_heatmap = cv2.resize(Grad_cam_heatmap[0], display_size)
                    reference_image = np.uint8(cv2.resize(img_array[0] * 255, display_size))
                    # Apply the CLAHE algorithm to the reference image
                    reference_image_CLAHE = np.GUIp(clahe.apply(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)), 0, 255)
                    # Display the heatmap in a new window
                    cv2.imshow('Grad-CAM Heatmap', Grad_cam_heatmap)
                    cv2.imshow('Reference Original Image', reference_image)
                    cv2.imshow('Reference Original Image (CLAHE)', reference_image_CLAHE)
                    cv2.waitKey(0)  # Wait for any key to be pressed
                    cv2.destroyAllWindows() # Close the window
    else:
        print_Color('~*ERROR: ~*image data doesnt exist.',
                    ['red', 'yellow'], advanced_mode=True)

# CI_rlmw
def CI_rlmw():
    # global var import
    global model
    # main proc
    model = None
    print_Color('loading the Ai model...', ['normal'])
    try:
        model = load_model(Model_dir)
    except (ImportError, IOError):
        print_Color('~*ERROR: ~*Failed to load the model. Try running `uaim` first.',
                    ['red', 'yellow'], advanced_mode=True)
    print_Color('loading the Ai model done.', ['normal'])

# CI_liid
def CI_liid(Auto: bool = False):
    # global var import
    global img_array
    global label
    replace_img = 'y'
    # check for img
    if img_array is not None and not Auto:
        # Ask the user if they want to replace the image
        print_Color('~*Warning: An image is already loaded. Do you want to replace it? ~*[~*Y~*/~*n~*]: ',
                    ['yellow', 'normal', 'green', 'normal', 'red', 'normal'],
                    advanced_mode=True,
                    print_END='')
        replace_img = input('')
        # If the user answers 'n' or 'N', return the existing img_array
    if replace_img.lower() == 'y':
        if not Auto:
            print_Color('img dir. Enter \'G\' for using GUI: ',
                        ['yellow'], print_END='')
            img_dir = input().strip('"')
            if img_dir.lower() == 'g':
                img_dir = open_file_GUI()
        else:
            img_dir = open_file_GUI()
        logger.debug(f'CI_liid:img_dir {img_dir}')
        # Extract file extension from img_dir
        try:
            _, file_extension = os.path.splitext(img_dir)
        except TypeError:
            file_extension = 'TEMP FILE EXTENSION'
        if file_extension.upper()[1:] not in IMG_AF:
            print_Color('~*ERROR: ~*Invalid file format. Please provide an image file.', ['red', 'yellow'],
                        advanced_mode=True)
            logger.warning('CI_liid>>ERROR: Invalid file format. Please provide an image file.')
        else:
            try:
                # Load and resize the image
                img = Image.open(img_dir).resize((IMG_RES[1], IMG_RES[0]))
            except Exception:
                print_Color('~*ERROR: ~*Invalid file dir. Please provide an image file.', ['red', 'yellow'],
                            advanced_mode=True)
                logger.warning('CI_liid>>ERROR: Invalid file dir. Please provide an image file.')
            else:
                # Check for RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Convert to numpy array
                img_array = np.asarray(img)

                # Normalize pixel values to [0, 1]
                img_array = img_array / 255.0

                # Add a dimension to transform from (height, width, channels) to (batch_size, height, width, channels)
                img_array = np.expand_dims(img_array, axis=0)

                # Assign labels to the image
                if not Auto:
                    print_Color('~*Enter label ~*(0 for Normal, 1 for Pneumonia, 2 Unknown): ', [
                                'yellow', 'normal'], print_END='', advanced_mode=True)
                    try:
                        label = int(input(''))
                    except ValueError:
                        print_Color('~*ERROR: ~*Invalid input.',
                                    ['red', 'yellow'], advanced_mode=True)
                        logger.warning('CI_liid>>ERROR: Invalid input label.')
                    else:
                        logger.debug(f'CI_liid:(INPUT) label {label}')
                        if label in [0, 1]:
                            # Convert label to categorical format
                            label = to_categorical(int(label), num_classes=2)
                            print_Color('The label is saved.', ['green'])
                        else:
                            label = None
                        print_Color('The image is loaded.', ['green'])

# CI_csaa
def CI_csaa():
    print_Color(CSAA, ['yellow', 'green'], advanced_mode=True)
    
# CI_uaim
def CI_uaim():
    print_Color('~*Do you want to download the light model? ~*[~*Y~*/~*n~*]: ',
            ['yellow', 'normal', 'green', 'normal', 'red', 'normal'],
            advanced_mode=True,
            print_END='')
    download_light_model = input('')
    if download_light_model.lower() == 'y':
        Github_repo_Releases_Model_name_temp = Github_repo_Releases_Model_light_name
    else:
        Github_repo_Releases_Model_name_temp = Github_repo_Releases_Model_name
    try:
        download_file_from_github(Github_repo_Releases_URL,
                                Github_repo_Releases_Model_name_temp,
                                Model_dir,
                                1024)
    except Exception:
        print_Color('\n~*ERROR: ~*Failed to download the model.', ['red', 'yellow'], advanced_mode=True)

# funcs(INTERNAL)>>>
# IEH
def IEH(id: str = 'Unknown', stop: bool = True, DEV: bool = True):
    print_Color(f'~*ERROR: ~*Internal error info/id:\n~*{id}~*.', ['red', 'yellow', 'bg_red', 'yellow'],
                advanced_mode=True)
    logger.exception(f'Internal Error Handler [stop:{stop}|DEV:{DEV}|id:{id}]')
    if DEV:
        print_Color('~*Do you want to see the detailed error message? ~*[~*Y~*/~*n~*]: ',
                    ['yellow', 'normal', 'green', 'normal', 'red', 'normal'],
                    advanced_mode=True,
                    print_END='')
        show_detailed_error = input('')
        if show_detailed_error.lower() == 'y':
            print_Color('detailed error message:', ['yellow'])
            traceback.print_exc()
    if stop:
        logger.warning('SYS EXIT|ERROR: Internal|by Internal Error Handler')
        sys.exit('SYS EXIT|ERROR: Internal|by Internal Error Handler')

# main
def main():
    # Text print
    print_Color(
        GUI_text_logo,
        ['yellow', 'green'],
        advanced_mode=True
    )
    # Making the GUI layout
    GUI_layout = [
        
    ]
    # GUI loop
    while True:
        

# start>>>
# clear the 'start L1' prompt
print('                  ', end='\r')
# Print CSAA
if SHOW_CSAA_OS:
    print_Color(CSAA, ['yellow', 'green'], advanced_mode=True)
# Start INFO
VER = f'V{GUI_Ver}' + datetime.now().strftime(" CDT(%Y/%m/%d | %H:%M:%S)")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    TF_MODE = 'GPU'
    TF_sys_details = tf.sysconfig.get_build_info()
    TF_CUDA_VER = TF_sys_details['cuda_version']
    TF_CUDNN_VER = TF_sys_details['cudnn_version']  # NOT USED
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "-L"]).decode("utf-8").split(":")[1].split("(")[0].strip()
        # GPU 0: NVIDIA `THE GPU NAME` (UUID: GPU-'xxxxxxxxxxxxxxxxxxxx')
        #     │                       │
        # ┌---┴----------┐        ┌---┴----------┐
        # │.split(":")[1]│        │.split("(")[0]│
        # └--------------┘        └--------------┘
    except Exception:
        gpu_name = '\x1b[0;31mNVIDIA-SMI-ERROR\x1b[0m'
    TF_INFO = f'GPU NAME: {gpus[0].name}>>{gpu_name}, CUDA Version: {TF_CUDA_VER}'
else:
    TF_MODE = 'CPU'
    info = cpuinfo.get_cpu_info()['brand_raw']
    TF_INFO = f'{info}'
# GUI_Info
GUI_Info = f'PDAI Ver: {VER} \nPython Ver: {sys.version} \nTensorflow Ver: {tf.version.VERSION}, Mode: {TF_MODE}, {TF_INFO}'
logger.info(f'PDAI Ver: {VER}')
logger.info(f'Python Ver: {sys.version}')
logger.info(f'Tensorflow Ver: {tf.version.VERSION}')
logger.info(f'Mode: {TF_MODE}, {TF_INFO}')
print(GUI_Info)
# FP
if Model_FORMAT not in ['TF_dir', 'H5_SF']:
    logger.info(f'Model file format [{Model_FORMAT}]')
    IEH(id=f'F[SYS],P[FP],Error[Invalid Model_FORMAT]', DEV=False)
elif Model_FORMAT == 'H5_SF':
    Model_dir += '.h5'
# start main
if __name__ == '__main__':
    try:
        try:
            main()
        except (EOFError, KeyboardInterrupt):
            logger.info('KeyboardInterrupt.')
            pass
    except Exception as e:
        IEH(id=f'F[SYS],RFunc[main],Error[{e}]', DEV=True)
    else:
        logger.info('GUI Exit.')
        print_Color('\n~*[PDAI GUI] ~*closed.',
                    ['yellow', 'red'], advanced_mode=True)
else:
    logger.info('GUI Imported.')
# end(EOF)
