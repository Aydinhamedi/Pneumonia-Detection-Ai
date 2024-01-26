import os
import hashlib
import json

def calculate_hash(file_path):
    with open(file_path, 'rb') as f:
        bytes = f.read()
        readable_hash = hashlib.sha256(bytes).hexdigest()
    return readable_hash

def check_file_type(file_name):
    if 'weight' in file_name.lower():
        return 'Weight'
    else:
        return 'Full'

def main():
    base_folder_path = 'models\Ready'
    model_info = {}
    for dir_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, dir_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                file_hash = calculate_hash(file_path)
                file_type = check_file_type(file_name)
                data = {
                    'name': file_name,
                    'Ver': dir_name,  
                    'stored_type': file_type
                }
                model_info[file_hash] = data

    with open('model_info.json', 'w') as json_file:
        json.dump(model_info, json_file)

if __name__ == "__main__":
    main()
