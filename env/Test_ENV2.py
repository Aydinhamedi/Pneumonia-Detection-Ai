# Utils
from Utils.one_cycle import OneCycleLr
from Utils.lr_find import LrFinder
from Utils.print_color_V2_NEW import print_Color_V2
from Utils.print_color_V1_OLD import print_Color
from Utils.Other import *

import PySimpleGUI as sg

def create_window(theme):
    # Set the theme
    sg.theme(theme)

    # Define the layout
    layout = [
        [sg.Text('Choose a Theme')],
        [sg.Combo(sg.theme_list(), key='-THEME-', enable_events=True, readonly=True)],
        [sg.Button('Apply Theme')],
        [sg.Text(size=(40,1), key='-OUTPUT-')]
    ]

    # Create the window
    return sg.Window('Theme Selector', layout)

# Create the initial window
window = create_window('Default')

# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == '-THEME-' or event == 'Apply Theme':
        theme = values['-THEME-']
        window.close()
        window = create_window(theme)

# Close the window
window.close()

window.close()
