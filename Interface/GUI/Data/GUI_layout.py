import PySimpleGUI as sg  
sg.theme('GrayGrayGray') 
# Making the GUI layout >>>
# Main
GUI_layout_Tab_main = [
    [sg.Text('Enter the image dir:', font=(None, 10, "bold"))],
    [
        sg.Input(key='-INPUT_IMG_dir-'),
        sg.Button('Browse', key='-BUTTON_BROWSE_IMG_dir-'),
        sg.Button('Ok', key='-BUTTON_OK_IMG_dir-')
    ],
    [sg.Text('Log:', font=(None, 10, "bold"))],
    [sg.Multiline(key='-OUTPUT_ST-', size=(54, 6), autoscroll=True)],
    [sg.Text('Result:', font=(None, 10, "bold"))],
    [sg.Text(key='-OUTPUT_ST_R-', size=(50, 2), background_color='white')],
    [
        sg.Checkbox('Show Grad-CAM', key='-CHECKBOX_SHOW_Grad-CAM-', default=True),
        sg.Checkbox('Show DICOM Info', key='-CHECKBOX_SHOW_DICOM_INFO-', default=True)
    ],
    [
        sg.Button('Analyse'),
        sg.Button('Close')
    ]
]
# Other
GUI_layout_Tab_other = [
    [sg.Text('Ai Model Settings:', font=(None, 10, "bold"))],
    [
        sg.Button('Update/Download Model', key='-BUTTON_UPDATE_MODEL-'),
        sg.Button('Reload Model', key='-BUTTON_RELOAD_MODEL-')
    ],
    [
        sg.Checkbox('Download Light Model', key='-CHECKBOX_DOWNLOAD_LIGHT_MODEL-', default=False)
    ],
    [sg.Text('Ai Model Info:', font=(None, 10, "bold"))],
    [
        sg.Text(key='-OUTPUT_Model_info-', size=(40, 7), pad=(4, 0))
    ]
]
# DICOM Info
def C_GUI_layout_DICOM_Info_Window():
    return [[sg.Multiline(key='-OUTPUT_DICOM_Info-', size=(100, 42), autoscroll=True)]]