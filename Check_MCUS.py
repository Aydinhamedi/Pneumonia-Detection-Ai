import hashlib
import sys

def calculate_hash(file_path):
    with open(file_path, 'rb') as file:
        bytes = file.read()
        readable_hash = hashlib.sha256(bytes).hexdigest()
    return readable_hash

def compare_files(file1, file2):
    file1_hash = calculate_hash(file1)
    file2_hash = calculate_hash(file2)

    if file1_hash == file2_hash:
        print(f"The files {file1} and {file2} are identical.")
    else:
        print(f"The files {file1} and {file2} are different.")
        sys.exit(1)

# Replace with your file paths
file1 = "Model_T&T.ipynb"
file2 = "BETA_E_Model_T&T.ipynb"

compare_files(file1, file2)
