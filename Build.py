# Pylib
import re
import os
import sys
import uuid
import shutil
import pprint
import py_compile
import subprocess as subP

# prep
pprint = pprint.PrettyPrinter(indent=4)


# Other funcs
def should_ignore(path, ignore_list):
    """Checks if a path should be ignored based on the provided ignore patterns.

    Args:
        path (str): The path to check.
        ignore_list (List[str]): The list of ignore patterns.

    Returns:
        bool: True if the path should be ignored, False otherwise.
    """
    for pattern in ignore_list:
        if re.search(pattern, path):
            return True
    return False


def copy_with_ignore(src, dst, ignore_list):
    """Recursively copies files from src to dst, ignoring files that match the ignore patterns.

    Args:
        src: Source directory path.
        dst: Destination directory path.
        ignore_list: List of glob patterns to ignore when copying.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_with_ignore(s, d, ignore_list)
        else:
            if not should_ignore(s, ignore_list):
                shutil.copy2(s, d)


def move_folders(src_dir, dest_dir):
    """Moves all subdirectories from a source directory to a destination directory.

    Args:
        src_dir (str): The source directory path.
        dest_dir (str): The destination directory path.
    """
    # Check if destination directory exists, if not create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all directories in the source directory
    for dir_name in os.listdir(src_dir):
        full_dir_name = os.path.join(src_dir, dir_name)

        # Check if it's a directory and not the destination directory
        if os.path.isdir(full_dir_name) and full_dir_name != dest_dir:
            # Move the directory to the destination directory
            shutil.move(full_dir_name, dest_dir)


def Compile_python(path):
    """Compiles python"""
    # Iterate over all files in the directory
    for root, dirs, files in os.walk(path):
        for filename in files:
            # Check if the file is a Python script
            if filename.endswith(".py"):
                # Compile the Python script to a .pyc file
                try:
                    py_compile.compile(os.path.join(root, filename), optimize=2)
                except py_compile.PyCompileError as e:
                    print(f"Failed to compile {filename}: {e}")
                    continue

                # Define the source file and destination file
                src = os.path.join(
                    root, "__pycache__", filename[:-3] + f".cpython-{sys.version_info.major}{sys.version_info.minor}.opt-2.pyc"
                )
                dst = os.path.join(root, filename[:-3] + ".pyc")

                # Check if the .pyc file exists
                if not os.path.exists(src):
                    print(src)
                    print(f"Failed to find .pyc file for {filename}")
                    continue

                # Move the .pyc file
                try:
                    shutil.move(src, dst)
                except shutil.Error as e:
                    print(f"Failed to move .pyc file for {filename}: {e}")
                    continue

                # Delete the original .py file
                try:
                    os.remove(os.path.join(root, filename))
                except OSError as e:
                    print(f"Failed to delete .py file for {filename}: {e}")


# Build funcs
def Build_Main():
    # Init
    RUN_Path_dict = {
        "Model info": ("python", "Make_model_info.py"),
        "Ruff (Check + Format)": ("Ruff_Auto.cmd"),
        "Sync code": ("Update_Code.cmd"),
        "Gen Requirements": ("Create_requirements.cmd"),
    }
    Sync_Utils = False
    # Starting...
    print("<Build Main> --> Starting...")
    # Proc Auto
    print("<Build Main> <Proc Auto - Init> --> Run dict:")
    for key, value in RUN_Path_dict.items():
        print(f" -- Key: {key}, Value: {value}")
    print("<Build Main> <Proc Auto - Start> --> Run dict:")
    for process in RUN_Path_dict:
        print(f"<Build Main> <Proc Auto> --> Running [{process}]...")
        # Run
        subP.run(RUN_Path_dict[process], shell=False)
        # End
        print(f"<Build Main> <Proc Auto> --> [{process}] Done.")
    # Sync Utils
    if Sync_Utils:
        print("<Build Main> <Sync Utils> --> Starting...")
        Main_utils_path = "Utils"
        utils_destinations = ["Interface\\GUI\\Data\\Utils", "Interface\\CLI\\Data\\Utils"]
        for utils_destination in utils_destinations:
            print(f"<Build Main> <Sync Utils> --> copying utils from {Main_utils_path} to {utils_destination}...")
            shutil.copytree(Main_utils_path, utils_destination, dirs_exist_ok=True)
        print("<Build Main> <Sync Utils> --> Done.")
    # Copy CLI / GUI Build
    print("<Build Main> <(CLI / GUI) Build> --> Starting...")
    Ignore_list = [
        r".*\.h5$",
        r".*\.pyc$",
        r".*\.tmp$",
        r".*\\*logs\\.*$",
        r".*\\__pycache__$",
        r".*\\GUI_Build\.py$",
        r".*\\GUI_DEV\.cmd$",
        r".*\\Data\\Gen_lib\.cmd$",
        r".*\\model_info\.json$",
    ]
    # CLI
    CLI_dir = "Interface\\CLI"
    CLI_Build_folder = "Build\\Github\\Releases\\Other\\Interface\\CLI"
    CLI_Archive_dir = "Build\\Github\\Releases\\Other\\Interface\\CLI\\Archive"
    CLI_Build_dir = f"Build\\Github\\Releases\\Other\\Interface\\CLI\\{uuid.uuid4()}"
    print("<Build Main> <(CLI / GUI) Build> --> CLI Build...")
    print(f" -- Build dir: {CLI_Build_dir}")
    print(" -- Archiving old builds...")
    move_folders(CLI_Build_folder, CLI_Archive_dir)
    print(" -- Copying new build...")
    copy_with_ignore(CLI_dir, CLI_Build_dir, Ignore_list)
    print("<Build Main> <(CLI / GUI) Build> --> CLI Build Done.")
    # GUI
    GUI_dir = "Interface\\GUI"
    GUI_Build_folder = "Build\\Github\\Releases\\Other\\Interface\\GUI"
    GUI_Archive_dir = "Build\\Github\\Releases\\Other\\Interface\\GUI\\Archive"
    GUI_Build_dir = f"Build\\Github\\Releases\\Other\\Interface\\GUI\\{uuid.uuid4()}"
    print("<Build Main> <(CLI / GUI) Build> --> GUI Build...")
    print(f" -- Build dir: {GUI_Build_dir}")
    print(" -- Archiving old builds...")
    move_folders(GUI_Build_folder, GUI_Archive_dir)
    print(" -- Copying new build...")
    copy_with_ignore(GUI_dir, GUI_Build_dir, Ignore_list)
    if input("<Build Main> <(CLI / GUI) Build> --> [Beta] Compile GUI? (y/n): ") == "y":
        print("<Build Main> <(CLI / GUI) Build> --> Compiling GUI...")
        Compile_python(GUI_Build_dir)
        print("<Build Main> <(CLI / GUI) Build> --> Compiling GUI Done.")
    print("<Build Main> <(CLI / GUI) Build> --> GUI Build Done.")
    print("<Build Main> <(CLI / GUI) Build> --> Done.")
    # End.
    print("<Build Main> --> End.")


# Main
def main():
    print("Starting the build... \n")
    Build_Main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nError: ")
        raise
    else:
        print("\nBuild complete.")
