# Libs
import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import glob 
import keras
import pprint
import random
import shutil
import gzip
import glob
import pickle
import datetime
import subprocess
import gpu_control
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from hyperas import optim
import tensorflow_addons as tfa
from importlib import reload
from keras.losses import categorical_crossentropy
import tensorflow as tf
from keras.models import Model
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from model_profiler import model_profiler
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, Nadam, RMSprop
from tensorflow_addons.optimizers import Yogi
from adabelief_tf import AdaBeliefOptimizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from keras.regularizers import l2
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from keras import Sequential
from random import randint, choice, shuffle
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,\
    MaxPooling2D,\
        Flatten,\
            Dense,\
                Dropout,\
                    BatchNormalization,\
                        SeparableConv2D,\
                            Input, Concatenate,\
                                GlobalAveragePooling2D,\
                                    CuDNNLSTM, concatenate,\
                                        Reshape, Multiply
# Utils
from Utils.one_cycle import OneCycleLr
from Utils.lr_find import LrFinder
from Utils.print_color_V2_NEW import print_Color_V2
from Utils.print_color_V1_OLD import print_Color
from Utils.Other import *
# Other
tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
# Main
from keras_gradient_noise import add_gradient_noise

# ...

NoisyAdam = add_gradient_noise(Adam)

