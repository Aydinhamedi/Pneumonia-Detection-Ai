#start L1
print('Loading the CLI...', end='\r')
#pylib
import re
import sys
import difflib
import inspect
import traceback
import subprocess
import cpuinfo
from loguru import logger
import efficientnet.tfkeras
from tkinter import filedialog
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
from PrintColor.Print_color import print_Color
#global vars>>>
#CONST SYS
CLI_Ver = '0.98'
Model_dir = 'Data/PAI_model' # without file extention
Database_dir = 'Data/dataset.npy'
IMG_AF = ('JPEG', 'PNG', 'BMP', 'TIFF', 'JPG')
Model_FORMAT = 'H5_SF' # TF_dir/H5_SF
IMG_RES = (300, 280, 3)
train_epochs_def = 4
SHOW_CSAA_OS = False
#normal global
img_array = None
Debug_m = False
label = None
model = None
#other
logger.remove()
logger.add('Data\\logs\\SYS_LOG_{time}.log', backtrace=True, diagnose=True, compression='zip')
logger.info('CLI Start...\n')
#crator
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
#HF>>>
#check_args
def check_arg(arg_list: list, arg_str: str, return_arg: bool = False, bool_OUTPUT_ONLY: bool = False):
    """
    This function checks if a specific argument exists in a list of arguments.

    Parameters:
    arg_list (list): A list of arguments.
    arg_str (str): The argument to check for.
    return_arg (bool, optional): If True, returns the string after the argument if it exists. Defaults to False.

    Returns:
    bool/str: Returns True if the argument exists and return_arg is False. 
              Returns the string after the argument if return_arg is True and the argument exists.
              Returns specific error codes in case of errors.
              
    Error Codes:

        '![IER:01]': This error is returned when the provided argument list (arg_list) is empty or contains only 'none' or ''.
                     It indicates that there are no arguments to check against.

        '![IER:02]': This error is returned when the argument to check for (arg_str) is an empty string.
                    It indicates that there is no argument specified to look for in the argument list.

        '![IER:03]': This error is returned when the argument to check for (arg_str) is found in the argument list (arg_list), 
                     but there is no string after the argument and return_arg is set to True.
                     It indicates that the function was expected to return the string following the argument, but there was none.

        '![IER:04]': This error is returned when the argument to check for (arg_str) is not found in the argument list (arg_list).
                     It indicates that the specified argument does not exist in the provided argument list.

        Note: If the bool_OUTPUT_ONLY parameter is set to True, the function will return False instead of these error codes.
    """

    # Error handling
    if arg_list == [] or arg_list == ['none'] or arg_list == ['']:
        return False if bool_OUTPUT_ONLY else '![IER:01]'
    if arg_str == '':
        return False if bool_OUTPUT_ONLY else '![IER:02]'
    
    for item in arg_list:
        if item.startswith('-'):
            if item[1] == arg_str:
                if len(item) == 2 and return_arg:
                    return False if bool_OUTPUT_ONLY else '![IER:03]'
                return True if not return_arg else item[2:]
    
    return False if bool_OUTPUT_ONLY else '![IER:04]'
check_arg_ERROR_LIST_USAGE = ['![IER:02]']
check_arg_ERROR_LIST_RT = ['![IER:04]', '![IER:03]']
#open_file_GUI
def open_file_GUI():
    formats = ";*.".join(IMG_AF)
    formats = "*." + formats.lower()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", formats)])
    if file_path:
        return file_path
#Debug
def Debug(ID, DEBUG_IF, SFL: bool = True, Force: bool = False, SFCS: bool = True):
    """
    This function is used for debugging purposes. It prints out various information about the data passed to it.

    Args:
        ID (Any): The identifier for the data. This could be any type, but is typically a string.
        DEBUG_IF (Any): The data that needs to be debugged. This could be any type.
        SFL (bool, optional): A flag to determine if the stack frame location should be included in the debug information. Defaults to True.
        Force (bool, optional): A flag to force the debug information to be printed even if the global Debug_m is set to False. Defaults to False.
        SFCS (bool, optional): A flag to determine if the function call stack should be included in the debug information. Defaults to True.

    Returns:
        None
    """
    try:
        if Debug_m or Force:
            frame_info = inspect.currentframe()
            stack_trace = traceback.format_stack()
            stack_trace_formated = ''
            for line in stack_trace[:-1]:
                stack_trace_formated += '--> [!>>>' + line 
            location = f'{inspect.stack()[1].filename}:{frame_info.f_back.f_lineno}' if SFL else f'L:{frame_info.f_back.f_lineno}'
            Debug_data = \
            f'\n~*--> ~*DEBUG INFO id: ~*[{str(ID)}]~*, ' \
            f'Location: ~*[{location}]~*, ' \
            f'time: ~*[{datetime.now().strftime("%Y/%m/%d | %H:%M:%S")}]\n~*--> ~*' \
            f'Data: ~*{str(DEBUG_IF)}\n~*--> ~*' \
            f'Data Type: ~*{type(DEBUG_IF)}\n~*--> ~*' \
            f'Memory Address: ~*DEC>>>~*{id(DEBUG_IF)}~* | HEX>>>~*{hex(id(DEBUG_IF))}~* | BIN>>>~*{bin(id(DEBUG_IF))}\n' 
            if SFCS:
                Debug_data += f'~*--> ~*Function Call Stack: ~*↓\n~*{stack_trace_formated}\n'
            print_Color(Debug_data,
            ['red', 'magenta', 'green', 'magenta', 'yellow', 'magenta', 'yellow',
            'red', 'magenta', 'yellow', 'red', 'magenta', 'yellow', 'red', 'magenta',
            'cyan', 'yellow', 'cyan', 'yellow', 'cyan', 'yellow', 'red', 'magenta', 'green', 'yellow'] if SFCS else \
            ['red', 'magenta', 'green', 'magenta', 'yellow', 'magenta', 'yellow',
            'red', 'magenta', 'yellow', 'red', 'magenta', 'yellow', 'red', 'magenta',
            'cyan', 'yellow', 'cyan', 'yellow', 'cyan', 'yellow'], 
            advanced_mode=True)
    except NameError:
        print_Color('~*[`Debug` func] --> ERROR: ~*carate a global var named `Debug_m` for turning on and off the Debug func.', ['red', 'yellow'], advanced_mode=True)
#CF>>>
#CI_help
def CI_help(SSUH: bool = True, show_lines: bool = True): #change show_lines and SSUH to change the style
    #main
    if SSUH:
        print_Color(f'{("┌─ " if show_lines else "")}~*Main (you can run them in order for simple usage):', ['cyan'], advanced_mode=True)
        for i, (cmd, desc) in enumerate(cmd_descriptions.items(), start=1):
            if i == len(cmd_descriptions):
                print_Color(f'{("│  └─ " if show_lines else "")}~*{i}. {cmd}: ~*{desc}', ['yellow', 'normal'], advanced_mode=True)
            else:
                print_Color(f'{("│  ├─ " if show_lines else "")}~*{i}. {cmd}: ~*{desc}', ['yellow', 'normal'], advanced_mode=True)
        #other
        print_Color(f'{("└─ " if show_lines else "")}~*Other:', ['cyan'], advanced_mode=True)
        for i, (cmd_other, desc_other) in enumerate(cmd_descriptions_other.items(), start=1):
            if i == len(cmd_descriptions_other):
                print_Color(f'{("   └─ " if show_lines else "")}~*{cmd_other}: ~*{desc_other}', ['yellow', 'normal'], advanced_mode=True)
            else:
                print_Color(f'{("   ├─ " if show_lines else "")}~*{cmd_other}: ~*{desc_other}', ['yellow', 'normal'], advanced_mode=True)
    else:
        print_Color(f'~*commands:', ['cyan'], advanced_mode=True)
        #main
        for i, (cmd, desc) in enumerate(cmd_descriptions.items(), start=1):
            if i == len(cmd_descriptions):
                print_Color(f'{("└─ " if show_lines else "")}~*{cmd}: ~*{desc}', ['yellow', 'normal'], advanced_mode=True)
            else:
                print_Color(f'{("├─ " if show_lines else "")}~*{cmd}: ~*{desc}', ['yellow', 'normal'], advanced_mode=True)
        #others
        for i, (cmd_other, desc_other) in enumerate(cmd_descriptions_other.items(), start=1):
            if i == len(cmd_descriptions_other):
                print_Color(f'{("└─ " if show_lines else "")}~*{cmd_other}: ~*{desc_other}', ['yellow', 'normal'], advanced_mode=True)
            else:
                print_Color(f'{("├─ " if show_lines else "")}~*{cmd_other}: ~*{desc_other}', ['yellow', 'normal'], advanced_mode=True)
#CI_atmd
def CI_atmd():
    #global var import
    global img_array
    global label
    #check for a image with a label
    if label is not None:
        # Check if the dataset file exists
        if os.path.exists(Database_dir):
            # Load the dataset file
            print_Color('loading the existing dataset...', ['normal'])
            dataset = np.load(Database_dir, allow_pickle=True).item()
        else:
            # Create a new dataset file if it doesn't exist
            dataset = {'images': [], 'labels': []}

        # Add the image array to the dataset
        dataset['images'].append(img_array)
        dataset['labels'].append(label)
        label_UF = np.argmax(label)
        label_class = 'PNEUMONIA' if label_UF == 1 else 'NORMAL'
        label_class_color = 'red' if label_UF == 1 else 'green'
        # Save the dataset file
        np.save(Database_dir, dataset)
        # Display the length of the dataset
        print(f"Dataset length: {len(dataset['images'])}")
        print_Color(f"Saved label: ~*{label_class}", [label_class_color], advanced_mode=True)
        print_Color('The image and its label are saved.', ['green'])
        label = None
    else:
        print_Color('~*ERROR: ~*a image with a label doesnt exist.', ['red', 'yellow'], advanced_mode=True)
#CI_tmwd
def CI_tmwd(argv_Split: list = ['none']):
    Debug('FUNC[CI_tmwd] ARGV INPUT', argv_Split)
    #global var import
    global model
    #argv
    train_epochs = check_arg(argv_Split, 'e', return_arg=True)
    Debug('FUNC[CI_tmwd] check_arg `-e`', train_epochs)
    if train_epochs in check_arg_ERROR_LIST_USAGE:
        IEH('Func[main>>CI_tmwd],P:[check_arg]>>[get `-e`],Error[check_arg.error in check_arg_ERROR_LIST_USAGE]', DEV=False)
    if train_epochs in check_arg_ERROR_LIST_RT or train_epochs.isalpha():
        print_Color(f'~*WARNING: ~*Invalid arg for -e. Using default value {train_epochs_def}.', ['red', 'yellow'], advanced_mode=True)
        train_epochs = train_epochs_def
    elif train_epochs == '![IER:01]':
        train_epochs = train_epochs_def
    train_epochs = int(train_epochs)
    # check the dataset file
    if os.path.exists(Database_dir):
        # Load the dataset file
        dataset = np.load(Database_dir, allow_pickle=True).item()
        if len(dataset['images']) > 15 or check_arg(argv_Split, 'i', bool_OUTPUT_ONLY=True): #ARG IL (ignore limits)
            # Convert 'dataset['images']' and 'dataset['labels']' to NumPy arrays
            images = np.array(dataset['images'])
            labels = np.array(dataset['labels'])
            images = np.reshape(images, (-1, IMG_RES[0], IMG_RES[1], IMG_RES[2]))
            try:
                if model is None:
                    print_Color('loading the Ai model...', ['normal'])
                    model = load_model(Model_dir)
            except (ImportError, IOError):
                print_Color('~*ERROR: ~*Failed to load the model.', ['red', 'yellow'], advanced_mode=True)
            else:
                print('Training the model...\n')
                #training
                history = model.fit(images, labels, epochs = train_epochs, batch_size = 1, verbose = 'auto')  # history not used
                print('Training done.\n')
        else:
            print_Color('~*ERROR: ~*Data/dataset.npy Len is <= 15 add more data.', ['red', 'yellow'], advanced_mode=True)
    else:
        print_Color('~*ERROR: ~*Data/dataset.npy doesnt exist.', ['red', 'yellow'], advanced_mode=True)
#CI_ulmd      
def CI_ulmd():
    print_Color('Warning: upload model data set (currently not available!!!)', ['yellow'])
#CI_pwai
def CI_pwai():
    #global var import
    global model
    #check for input img
    if img_array is not None:
        try:
            if model is None:
                print_Color('loading the Ai model...', ['normal'])
                model = load_model(Model_dir)
        except (ImportError, IOError):
            print_Color('~*ERROR: ~*Failed to load the model.', ['red', 'yellow'], advanced_mode=True)
        else:
            print_Color('predicting with the Ai model...', ['normal'])
            model_prediction_ORG = model.predict(img_array)
            model_prediction = np.argmax(model_prediction_ORG, axis=1)
            pred_class = 'PNEUMONIA' if model_prediction == 1 else 'NORMAL'
            class_color = 'red' if model_prediction == 1 else 'green'
            confidence = np.max(model_prediction_ORG)
            print_Color(f'~*the Ai model prediction: ~*{pred_class}~* with confidence ~*{confidence:.2f}~*.', ['normal', class_color, 'normal', 'green', 'normal'], advanced_mode=True)
            if confidence < 0.82:
                print_Color('~*WARNING: ~*the confidence is low.', ['red', 'yellow'], advanced_mode=True)
    else:
        print_Color('~*ERROR: ~*image data doesnt exist.', ['red', 'yellow'], advanced_mode=True)
#CI_rlmw
def CI_rlmw():
    #global var import
    global model
    #main proc
    model = None
    print_Color('loading the Ai model...', ['normal'])
    try:
        model = load_model(Model_dir)
    except (ImportError, IOError):
        print_Color('~*ERROR: ~*Failed to load the model.', ['red', 'yellow'], advanced_mode=True)
    print_Color('loading the Ai model done.', ['normal'])
#CI_liid
def CI_liid():
    #global var import
    global img_array
    global label
    replace_img = 'y'
    #check for img
    if img_array is not None:
        # Ask the user if they want to replace the image
        print_Color('~*Warning: An image is already loaded. Do you want to replace it? ~*[~*Y~*/~*n~*]: ',
                    ['yellow', 'normal', 'green', 'normal', 'red', 'normal'],
                    advanced_mode = True,
                    print_END='')
        replace_img = input('')
        # If the user answers 'n' or 'N', return the existing img_array
    if replace_img.lower() == 'y':   
        print_Color('img dir. Enter \'G\' for using GUI: ', ['yellow'], print_END='')
        img_dir = input().strip('"')
        if img_dir.lower() == 'g':
            img_dir = open_file_GUI()
        
        logger.debug(f'liid:img_dir {img_dir}')
        # Extract file extension from img_dir
        _, file_extension = os.path.splitext(img_dir)
        # Check if file is an image of acceptable format
        if file_extension.upper()[1:] not in IMG_AF:
            print_Color('~*ERROR: ~*Invalid file format. Please provide an image file.', ['red', 'yellow'], advanced_mode=True)
        else:
            try:
                # Load and resize the image
                img = Image.open(img_dir).resize((IMG_RES[1], IMG_RES[0]))
            except Exception:
                print_Color('~*ERROR: ~*Invalid file dir. Please provide an image file.', ['red', 'yellow'], advanced_mode=True)
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
                print_Color('Enter label (0 for Normal, 1 for Pneumonia, 2 Unknown): ', ['yellow'], print_END='')
                try:
                    label = int(input(''))
                except ValueError:
                    print_Color('~*ERROR: ~*Invalid input.', ['red', 'yellow'], advanced_mode=True)
                else:
                    if label in [0, 1]:
                        # Convert label to categorical format
                        label = to_categorical(int(label), num_classes=2)
                        print_Color('The label is saved.', ['green'])
                    else:
                        label = None
                    print_Color('The image is loaded.', ['green'])
#CI_csaa
def CI_csaa():
    print_Color(CSAA, ['yellow', 'green'], advanced_mode=True)
#CMT>>>
command_tuple = (
    'help', # help
    'atmd', # add to model dataset
    'tmwd', # train model with dataset
    'ulmd', # upload model data set (not available!!!) 
    'pwai', # predict with Ai 
    'rlmw', # reload model
    'liid', # load img input data
    'csaa', # Creator Signature ASCII ART
    'debug', # Debug
    'exit', # Quit the CLI
    'clear' # Clear the CLI
)
#SCH table:
# '│' (U+2502): Box Drawings Light Vertical
# '┌' (U+250C): Box Drawings Light Down and Right
# '┐' (U+2510): Box Drawings Light Down and Left
# '└' (U+2514): Box Drawings Light Up and Right
# '┘' (U+2518): Box Drawings Light Up and Left
# '├' (U+251C): Box Drawings Light Vertical and Right
# '┤' (U+2524): Box Drawings Light Vertical and Left
# '┬' (U+252C): Box Drawings Light Down and Horizontal
# '┴' (U+2534): Box Drawings Light Up and Horizontal
# '┼' (U+253C): Box Drawings Light Vertical and Horizontal
# '─'
cmd_descriptions = {
    'help': 'Show the help menu with the list of all available commands',
    'liid': 'Load image data for input',
    'pwai': 'Make predictions using the trained AI model'
}
cmd_descriptions_other = {
    'atmd': 'Add data to the model dataset for training',
    'tmwd': f'Train the model with the existing dataset.\n\
   │  └────Optional Args:\n\
   │       ├────\'-i\' Ignore the limits.\n\
   │       └────\'-e\' The number after \'e\' will be training epochs (default: {train_epochs_def}).\n\
   │            └────Example: \'-e10\'',
    'ulmd': 'Upload model data set (currently not available)',
    'csaa': 'Creator Signature ASCII ART',
    'rlmw': 'Reload/Load Ai model',
    'exit': 'Quit the CLI',
    'clear': 'Clear the CLI'
}
#funcs(INTERNAL)>>>
#CLI_IM
def CLI_IM(CLII: bool = True):
    if CLII: print_Color('>>> ', ['green'], print_END='')  
    U_input = input('').lower()
    try:
        str_array = U_input.split()
        if str_array[0] in command_tuple:
            return str_array
        else:
            closest_match = difflib.get_close_matches(str_array[0], command_tuple, n=1)
            if closest_match:
                print_Color(f'~*ERROR: ~*Invalid input. you can use \'~*help~*\', did you mean \'~*{closest_match[0]}~*\'.', ['red', 'yellow', 'green', 'yellow', 'green', 'yellow'], advanced_mode=True)
            else:
                print_Color(f'~*ERROR: ~*Invalid input. you can use \'~*help~*\'.', ['red', 'yellow', 'green', 'yellow'], advanced_mode=True)
            return ['IIE']
    except IndexError:
        return ['IIE']
#IEH
def IEH(id: str = 'Unknown', stop: bool = True, DEV: bool = True):
    print_Color(f'~*ERROR: ~*Internal error info/id:\n~*{id}~*.', ['red', 'yellow', 'bg_red', 'yellow'], advanced_mode=True)   
    logger.exception(f'Internal Error Handler [stop:{stop}|id:{id}]')
    if DEV: 
        print_Color('~*Do you want to see the detailed error message? ~*[~*Y~*/~*n~*]: ',
                    ['yellow', 'normal', 'green', 'normal', 'red', 'normal'],
                    advanced_mode = True,
                    print_END='')
        show_detailed_error = input('')
        if show_detailed_error.lower() == 'y':
            print_Color('detailed error message:', ['yellow'])
            traceback.print_exc()
    if stop: sys.exit('SYS EXIT|ERROR: Internal|by Internal Error Handler')
#main
def main():
    #global
    global Debug_m
    #CLI loop
    while True: #WT
        #input manager
        input_array = CLI_IM()
        Debug('input_array', input_array)
        logger.debug(f'input_array {input_array}')
        Debug('Model_dir', Model_dir)
        match input_array[0]: #MI
            case 'help':
                CI_help() 
            case 'atmd':
                CI_atmd()  
            case 'tmwd':
                if len(input_array) > 1:
                    CI_tmwd(argv_Split = input_array[1:])  
                else:
                    CI_tmwd()    
            case 'ulmd':
                CI_ulmd()  
            case 'pwai': 
                CI_pwai()
            case 'rlmw': 
                CI_rlmw()
            case 'liid':
                CI_liid()
            case 'csaa':
                CI_csaa()
            case 'IIE':
                pass
            case 'debug':
                print('Debug mode is ON...')
                Debug_m = True
            case 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print(CLI_Info)
            case 'exit':
                logger.info('Exit by prompt.')
                raise KeyboardInterrupt
            case _:
                IEH(id = 'Func[main],P:[CLI loop]>>[match input],Error[nothing matched]', stop = False, DEV = False)
#start>>>
#clear the 'start L1' prompt
print('                  ', end='\r')
#Print CSAA
if SHOW_CSAA_OS: print_Color(CSAA, ['yellow', 'green'], advanced_mode=True)
#Start INFO
VER = f'V{CLI_Ver}' + datetime.now().strftime(" CDT(%Y/%m/%d | %H:%M:%S)")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    TF_MODE = 'GPU'
    TF_sys_details = tf.sysconfig.get_build_info()
    TF_CUDA_VER = TF_sys_details['cuda_version']
    TF_CUDNN_VER = TF_sys_details['cudnn_version'] #NOT USED
    try:
        gpu_name = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").split(":")[1].split("(")[0].strip()
        #GPU 0: NVIDIA `THE GPU NAME` (UUID: GPU-'xxxxxxxxxxxxxxxxxxxx')
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
#CLI_Info
CLI_Info = f'PDAI Ver: {VER} \nPython Ver: {sys.version} \nTensorflow Ver: {tf.version.VERSION}, Mode: {TF_MODE}, {TF_INFO} \nType \'help\' for more information.'
logger.info(f'PDAI Ver: {VER}')
logger.info(f'Python Ver: {sys.version}')
logger.info(f'Tensorflow Ver: {tf.version.VERSION}')
logger.info(f'Mode: {TF_MODE}, {TF_INFO}')
print(CLI_Info)
#FP
if Model_FORMAT not in ['TF_dir', 'H5_SF']:
    logger.info(f'Model file format [{Model_FORMAT}]')
    IEH(id=f'F[SYS],P[FP],Error[Invalid Model_FORMAT]', DEV=False)
elif Model_FORMAT == 'H5_SF':
    Model_dir += '.h5'
#start main
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
        logger.info('CLI Exit.')
        print_Color('\n~*[PDAI CLI] ~*closed.', ['yellow', 'red'], advanced_mode=True)
else:
    logger.info('CLI Imported.')
#end(EOF) 