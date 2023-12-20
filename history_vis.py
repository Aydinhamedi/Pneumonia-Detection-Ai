from Utils.Other import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# load history
history = load_list('history\\model_history.pkl.gz', compressed=True)

# Chunk size for 3D plot
chunk_size = 6  # Change this to your desired chunk size


def chunked_data(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


try:
    EPM = 'Epoch(Subset)'  

    # Calculate deltas
    delta_loss = np.diff(history['loss'])
    delta_accuracy = np.diff(history['accuracy'])

    try:
        delta_val_loss = np.diff(history['val_loss'])
        delta_val_accuracy = np.diff(history['val_accuracy'])
    except (ValueError, NameError):
        print('\033[91mfailed to load val_loss or val_accuracy for delta calculation.')

    plt.figure(figsize=(16, 10))
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='loss')
    try:
        plt.plot(history['val_loss'], label='val_loss', color='orange')
    except (ValueError, NameError):
        print('\033[91mfailed to load val_loss.')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel(EPM)
    plt.ylim(top=max(history['val_loss'][10:]), bottom=0) # (max(history['val_loss'][8:]) + min(history['val_loss'])) / 2
    plt.grid(True)
    
    # Density plot for loss
    plt.subplot(2, 2, 2)
    plt.hist(history['loss'], label='loss density', color='blue', alpha=0.5, bins=100)
    try:
        plt.hist(history['val_loss'], label='val_loss density', color='orange', alpha=0.5, bins=100)
    except (ValueError, NameError):
        print('\033[91mfailed to load val_loss (density plot).')
    plt.title('Density Plot for Loss')
    plt.xlabel('Loss')
    plt.xlim(right=max(history['val_loss'][10:])) # (max(history['val_loss'][8:]) + min(history['val_loss'])) / 2
    plt.grid(True)
    
    
    # Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history['accuracy'], label='accuracy')
    try:
        plt.plot(history['val_accuracy'], label='val_accuracy', color='orange')
    except (ValueError, NameError):
        print('\033[91mfailed to load val_accuracy.')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel(EPM)
    plt.grid(True)
    
    # Density plot for accuracy
    plt.subplot(2, 2, 4)
    plt.hist(history['accuracy'], label='accuracy density', color='blue', alpha=0.5, bins=40)
    try:
        plt.hist(history['val_accuracy'], label='val_accuracy density', color='orange', alpha=0.5, bins=40)
    except (ValueError, NameError):
        print('\033[91mfailed to load val_accuracy (density plot).')
    plt.title('Density Plot for Accuracy')
    plt.xlabel('Accuracy')
    plt.grid(True)

    # Delta Loss
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    plt.plot(delta_loss, label='delta_loss')
    try:
        plt.plot(delta_val_loss, label='delta_val_loss', color='orange')
    except (ValueError, NameError):
        print('\033[91mfailed to load delta_val_loss.')
    plt.title('Delta Model Loss')
    plt.ylabel('Delta Loss')
    plt.ylim(top=1.5, bottom=-1.5) 
    plt.xlabel(EPM)
    plt.grid(True)
    # Delta Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(delta_accuracy, label='delta_accuracy')
    try:
        plt.plot(delta_val_accuracy, label='delta_val_accuracy', color='orange')
    except (ValueError, NameError):
        print('\033[91mfailed to load delta_val_accuracy.')
    plt.title('Delta Model Accuracy')
    plt.ylabel('Delta Accuracy')
    plt.xlabel(EPM)
    plt.grid(True)

    # Calculate chunked data
    chunked_loss = chunked_data(history['val_loss'], chunk_size)
    chunked_accuracy = chunked_data(history['val_accuracy'], chunk_size)

    # Clip the loss values to a maximum of max(history['val_loss'][10:])
    max_loss = max(history['val_loss'][10:])
    chunked_loss = np.clip(chunked_loss, a_min=None, a_max=max_loss)

    # Create 3D surface plots for each chunk
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, projection='3d')
    X = np.arange(len(chunked_loss))
    Y = np.arange(chunk_size)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(chunked_loss).T  # Transpose the array to match the shape of X and Y
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('3D Surface Plot of Chunked Loss')
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('Loss')

    ax = fig.add_subplot(122, projection='3d')
    X = np.arange(len(chunked_accuracy))
    Y = np.arange(chunk_size)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(chunked_accuracy).T  # Transpose the array to match the shape of X and Y
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('3D Surface Plot of Chunked Accuracy')
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('Accuracy')

    plt.tight_layout()
    plt.show()

except (ValueError, NameError) as E:
    print(f'\033[91mFailed to load model history.\nError: {E}')