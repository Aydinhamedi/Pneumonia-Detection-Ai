import os
import hashlib
import json


def calculate_hash(file_path):
    """Calculates a SHA256 hash for the contents of the file at the given path.

    Args:
        file_path: The path to the file to hash.

    Returns:
        The hex string of the SHA256 hash.
    """
    with open(file_path, "rb") as f:
        bytes = f.read()
        readable_hash = hashlib.sha256(bytes).hexdigest()
    return readable_hash


def check_file_type(file_name):
    """Checks if a file name contains 'weight' to determine the file type.

    Args:
        file_name (str): The file name to check.

    Returns:
        str: 'Weight' if 'weight' is in the name, 'Full' otherwise.
    """
    if "weight" in file_name.lower():
        return "Weight"
    else:
        return "Full"


def main():
    """
    Generates a JSON file containing hashes and metadata for all model files in the 'models/Ready' folder.

    Iterates through all subfolders in 'models/Ready', gets file info for each model file,
    and saves it to a dict where the key is the file hash. Writes this model info dict
    to 'model_info.json'.
    """
    base_folder_path = "models\Ready"
    model_info = {}
    for dir_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, dir_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                file_hash = calculate_hash(file_path)
                file_type = check_file_type(file_name)
                data = {"Ver": dir_name, "stored_type": file_type}
                model_info[file_hash] = data

    with open("model_info.json", "w") as json_file:
        json.dump(model_info, json_file)


if __name__ == "__main__":
    main()
