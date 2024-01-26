# Copyright (c) 2023 Aydin Hamedi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# start L1
print('Loading the GUI...', end='\r')
# pylib
import os
import re
from time import sleep
import cv2
import sys
import queue
import cpuinfo
import difflib
import inspect
import traceback
import subprocess
import threading
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
GUI_Ver = '0.8.9.5 (GUI)'
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
class CustomQueue:
    # Custom queue class with size limit 
    #
    # Initializes a Queue instance with a max size. Provides put(), get(), 
    # and is_updated() methods to add items, retrieve items, and check if 
    # updated since last get() call.
    def __init__(self, max_items=4):
        self.q = queue.Queue()
        self.max_items = max_items
        self.is_updated = False

    def put(self, item):
        if self.q.qsize() == self.max_items:
            self.q.get()
        self.q.put(item)
        self.is_updated = True

    def get(self, reset_updated=True):
        items = list(self.q.queue)
        if reset_updated:
            self.is_updated = False
        return items

    def is_updated(self):
        return self.is_updated

Queue_ins = CustomQueue()
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
def CI_pwai(show_gradcam: bool = True) -> str:
    # global var import
    global model
    # check for input img
    if img_array is not None:
        try:
            if model is None:
                print_Color('loading the Ai model...', ['normal'])
                model = load_model(Model_dir)
        except (ImportError, IOError):
            return 'ERROR: Failed to load the model.'
        else:
            print_Color('predicting with the Ai model...', ['normal'])
            model_prediction_ORG = model.predict(img_array)
            model_prediction = np.argmax(model_prediction_ORG, axis=1)
            pred_class = 'PNEUMONIA' if model_prediction == 1 else 'NORMAL'
            confidence = np.max(model_prediction_ORG)
            return_temp = f'the Ai model prediction: {pred_class} with confidence {confidence:.2f}.'
            if confidence < 0.82:
                return_temp +='WARNING: the confidence is low.'           
            if model_prediction == 1 and show_gradcam:
                clahe = cv2.createCLAHE(clipLimit=1.8)
                Grad_cam_heatmap = make_gradcam_heatmap(img_array,
                                                        model, 'top_activation',
                                                        second_last_conv_layer_name = 'top_conv',
                                                        sensitivity_map = 2, pred_index=tf.argmax(model_prediction_ORG[0])) 
                Grad_cam_heatmap = cv2.resize(np.clip(Grad_cam_heatmap, 0, 1), (img_array.shape[1], img_array.shape[2]))
                Grad_cam_heatmap = np.uint8(255 * Grad_cam_heatmap)
                Grad_cam_heatmap = cv2.applyColorMap(Grad_cam_heatmap, cv2.COLORMAP_VIRIDIS)
                Grad_cam_heatmap = np.clip(np.uint8((Grad_cam_heatmap * 0.3) + ((img_array * 255) * 0.7)), 0, 255)
                # Resize the heatmap for a larger display
                display_size = (600, 600)  # Change this to your desired display size
                Grad_cam_heatmap = cv2.resize(Grad_cam_heatmap[0], display_size)
                reference_image = np.uint8(cv2.resize(img_array[0] * 255, display_size))
                # Apply the CLAHE algorithm to the reference image
                reference_image_CLAHE = np.clip(clahe.apply(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)), 0, 255)
                # Display the heatmap in a new window
                cv2.imshow('Grad-CAM Heatmap', Grad_cam_heatmap)
                cv2.imshow('Reference Original Image', reference_image)
                cv2.imshow('Reference Original Image (CLAHE)', reference_image_CLAHE)
            return return_temp 
    else:
        print_Color('~*ERROR: ~*image data doesnt exist.',
                    ['red', 'yellow'], advanced_mode=True, return_str=True)

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
def CI_liid(img_dir) -> str:
    # global var import
    global img_array
    # check for img
    logger.debug(f'CI_liid:img_dir {img_dir}')
    # Extract file extension from img_dir
    try:
        _, file_extension = os.path.splitext(img_dir)
    except TypeError:
        file_extension = 'TEMP FILE EXTENSION'
    if file_extension.upper()[1:] not in IMG_AF:
        logger.warning('CI_liid>>ERROR: Invalid file format. Please provide an image file.')
        return 'ERROR: Invalid file format. Please provide an image file.'    
    else:
        try:
            # Load and resize the image
            img = Image.open(img_dir).resize((IMG_RES[1], IMG_RES[0]))
        except Exception:
            logger.warning('CI_liid>>ERROR: Invalid file dir. Please provide an image file.')
            return 'ERROR: Invalid file dir. Please provide an image file.'
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
            
            return 'Image loaded.'

# CI_uaim
def CI_uaim(download_light_model: bool = False):
    if download_light_model:
        Log_temp_txt = 'Downloading the model...\n'
        Github_repo_Releases_Model_name_temp = Github_repo_Releases_Model_light_name
    else:
        Log_temp_txt = 'Downloading the light model...\n'
        Github_repo_Releases_Model_name_temp = Github_repo_Releases_Model_name
    Queue_ins.put(Log_temp_txt)
    try:
        download_file_from_github(Github_repo_Releases_URL,
                                Github_repo_Releases_Model_name_temp,
                                Model_dir,
                                1024)
        CI_rlmw()
        print('Model downloaded.')
    except Exception:
        Queue_ins.put('ERROR: Failed to download the model.')
    Queue_ins.put('Model downloaded.')
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
    # global
    global GUI_window
    # Text print
    print_Color(
        GUI_text_logo,
        ['yellow', 'green'],
        advanced_mode=True
    )
    # prep var
    IMG_dir = None
    # Making the GUI layout
    GUI_layout = [
        [sg.Text('Enter the image dir:')],
        [
            sg.Input(key='-INPUT_IMG_dir-'),
            sg.Button('Browse', key='-BUTTON_BROWSE_IMG_dir-'),
            sg.Button('Ok', key='-BUTTON_OK_IMG_dir-')
        ],
        [sg.Text('Log:')],
        [sg.Text(key='-OUTPUT_ST-', size=(48, 4))],
        [sg.Text('Result:')],
        [sg.Text(key='-OUTPUT_ST_R-', size=(48, 2))],
        [
            sg.Checkbox('Show Grad-CAM', key='-CHECKBOX_SHOW_Grad-CAM-', default=True),
            sg.Checkbox('Download Light Model', key='-CHECKBOX_DOWNLOAD_LIGHT_MODEL-', default=False)
        ],
        [
            sg.Button('Update/Download Model', key='-BUTTON_UPDATE_MODEL-'),
            sg.Button('Analyse'),
            sg.Button('Close')
        ]
    ]
    # Create the window
    GUI_window = sg.Window('Pneumonia-Detection-Ai-GUI', GUI_layout)
    # Main loop for the Graphical User Interface (GUI)
    while True:
        # Read events and values from the GUI window
        event, values = GUI_window.read(timeout=250, timeout_key='-TIMEOUT-')
        if not event == '-TIMEOUT-':
            logger.debug(f'GUI_window:event: {event}')
            logger.debug(f'GUI_window:values: {values}')
        
        # Check if the window has been closed or the 'Close' button has been clicked
        if event == sg.WINDOW_CLOSED or event == 'Close':
            break  # Exit the loop and close the window

        # Handle event for browsing and selecting an image directory
        if event == '-BUTTON_BROWSE_IMG_dir-':
            # Open file dialog to select an image, and update the input field with the selected directory
            IMG_dir = open_file_GUI()
            GUI_window['-INPUT_IMG_dir-'].update(IMG_dir)

        # Handle event for confirming the selected image directory
        if event == '-BUTTON_OK_IMG_dir-':
            # Retrieve the image directory from the input field and update the display
            IMG_dir = GUI_window['-INPUT_IMG_dir-'].get()
            GUI_window['-INPUT_IMG_dir-'].update(IMG_dir)

        # Handle event for analyzing the selected image
        if event == 'Analyse':
            # Call the function to load the image and update the output status
            Log_temp_txt = CI_liid(IMG_dir)
            Queue_ins.put(Log_temp_txt)

            # If the image is successfully loaded, proceed with analysis
            if Log_temp_txt == 'Image loaded.':
                Queue_ins.put('Analyzing...')

                # Call the function to perform pneumonia analysis and display the results
                Log_temp_txt2 = CI_pwai(show_gradcam=values['-CHECKBOX_SHOW_Grad-CAM-'])
                GUI_window['-OUTPUT_ST_R-'].update(
                    Log_temp_txt2,
                    text_color='green' if 'NORMAL' in Log_temp_txt2 else 'red',
                    background_color='white'
                )
                GUI_window.finalize()

        # Handle event for updating the AI model
        if event == '-BUTTON_UPDATE_MODEL-':
            # Start a new thread to download the model without freezing the GUI
            threading.Thread(
                target=CI_uaim,
                args=(values['-CHECKBOX_DOWNLOAD_LIGHT_MODEL-'],),
                daemon=True
            ).start()

        # Continuously check if there are results in the queue to be processed
        if Queue_ins.is_updated:
            # Retrieve the result from the queue
            result_expanded = ''
            result = Queue_ins.get()
            print(f'Queue Data: {result}')
            logger.debug(f'Queue:get: {result}')
            # Update the GUI with the result message
            for block in result:
                result_expanded += f'> {block}\n'
            GUI_window['-OUTPUT_ST-'].update(result_expanded, text_color='yellow')
            GUI_window.finalize()

# start>>>
# clear the 'start L1' prompt
print('                  ', end='\r')
# Print CSAA
if SHOW_CSAA_OS:
    print_Color(CSAA, ['yellow', 'green'], advanced_mode=True)
# Start INFO
VER = f'V{GUI_Ver}' + datetime.now().strftime(" | CDT(%Y/%m/%d | %H:%M:%S)")
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
