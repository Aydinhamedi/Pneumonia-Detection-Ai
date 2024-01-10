import subprocess
import traceback
import sys
import os
# Other
from Data.Utils.print_color_V1_OLD import print_Color
def run_program(file_path):
    while True:
        try:
            try:
                # Run the other Python program using subprocess
                subprocess.run(["python", file_path], check=True)
            except subprocess.CalledProcessError as ERROR_Py:
                print_Color("~*An error occurred: \nERROR: ~*" + str(ERROR_Py), ['yellow', 'red'], advanced_mode=True)
                print_Color('~*Do you want to see the detailed error message? ~*[~*Y~*/~*n~*]: ',
                        ['yellow', 'normal', 'green', 'normal', 'red', 'normal'],
                        advanced_mode = True,
                        print_END='')
                show_detailed_error = input('')
                if show_detailed_error.lower() == 'y':
                    print_Color('detailed error message:', ['yellow'])
                    #print_Color('1th ERROR (FILE ERROR) [!MAIN!]:', ['red'])
                    print_Color('2th ERROR (subprocess.run ERROR):', ['red'])
                    traceback.print_exc()
                choice = input("Do you want to restart the program? (y/n): ")
                if choice.lower() != "y":
                    break
                os.system('cls' if os.name == 'nt' else 'clear')
            else:
                break
        except OSError:
            break
run_program('Data\CLI_main.py')