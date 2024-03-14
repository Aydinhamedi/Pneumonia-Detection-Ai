from Utils.Other import * # noqa: F403
import matplotlib.pyplot as plt
import numpy as np

# load history
history = load_list("history\\model_history.pkl.gz", compressed=True) # noqa: F405

# Chunk size for 3D plot
chunk_size = 6  # Change this to your desired chunk size


def chunked_data(data, chunk_size):
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


try:
    EPM = "Epoch(Subset)"

    # Calculate deltas
    delta_loss = np.diff(history["loss"])
    delta_accuracy = np.diff(history["accuracy"])

    try:
        delta_val_loss = np.diff(history["val_loss"])
        delta_val_accuracy = np.diff(history["val_accuracy"])
    except (ValueError, NameError):
        print("\033[91mfailed to load val_loss or val_accuracy for delta calculation.")

    plt.figure(figsize=(16, 10))
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history["loss"], label="loss")
    try:
        plt.plot(history["val_loss"], label="val_loss", color="orange")
    except (ValueError, NameError):
        print("\033[91mfailed to load val_loss.")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel(EPM)
    plt.ylim(top=max(history["val_loss"][10:]), bottom=0)  # (max(history['val_loss'][8:]) + min(history['val_loss'])) / 2
    plt.grid(True)

    # Density plot for loss
    plt.subplot(2, 2, 2)
    plt.hist(history["loss"], label="loss density", color="blue", alpha=0.5, bins=100)
    try:
        plt.hist(
            history["val_loss"],
            label="val_loss density",
            color="orange",
            alpha=0.5,
            bins=100,
        )
    except (ValueError, NameError):
        print("\033[91mfailed to load val_loss (density plot).")
    plt.title("Density Plot for Loss")
    plt.xlabel("Loss")
    plt.xlim(right=max(history["val_loss"][10:]), left=0)  # (max(history['val_loss'][8:]) + min(history['val_loss'])) / 2
    plt.grid(True)

    # Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history["accuracy"], label="accuracy")
    try:
        plt.plot(history["val_accuracy"], label="val_accuracy", color="orange")
    except (ValueError, NameError):
        print("\033[91mfailed to load val_accuracy.")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel(EPM)
    plt.grid(True)

    # Density plot for accuracy
    plt.subplot(2, 2, 4)
    plt.hist(history["accuracy"], label="accuracy density", color="blue", alpha=0.5, bins=40)
    try:
        plt.hist(
            history["val_accuracy"],
            label="val_accuracy density",
            color="orange",
            alpha=0.5,
            bins=40,
        )
    except (ValueError, NameError):
        print("\033[91mfailed to load val_accuracy (density plot).")
    plt.title("Density Plot for Accuracy")
    plt.xlabel("Accuracy")
    plt.grid(True)

    # Delta Loss
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    plt.plot(delta_loss, label="delta_loss")
    try:
        plt.plot(delta_val_loss, label="delta_val_loss", color="orange")
    except (ValueError, NameError):
        print("\033[91mfailed to load delta_val_loss.")
    plt.title("Delta Model Loss")
    plt.ylabel("Delta Loss")
    plt.ylim(top=1.5, bottom=-1.5)
    plt.xlabel(EPM)
    plt.grid(True)
    # Delta Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(delta_accuracy, label="delta_accuracy")
    try:
        plt.plot(delta_val_accuracy, label="delta_val_accuracy", color="orange")
    except (ValueError, NameError):
        print("\033[91mfailed to load delta_val_accuracy.")
    plt.title("Delta Model Accuracy")
    plt.ylabel("Delta Accuracy")
    plt.xlabel(EPM)
    plt.grid(True)

    # Calculate chunked data
    chunked_loss = chunked_data(history["val_loss"], chunk_size)
    chunked_accuracy = chunked_data(history["val_accuracy"], chunk_size)

    # Clip the loss values to a maximum of max(history['val_loss'][10:])
    max_loss = max(history["val_loss"][10:])
    chunked_loss = np.clip(chunked_loss, a_min=None, a_max=max_loss)

    # Create 3D surface plots for each chunk
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, projection="3d")
    X = np.arange(len(chunked_loss))
    Y = np.arange(chunk_size)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(chunked_loss).T  # Transpose the array to match the shape of X and Y
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_title("3D Surface Plot of Chunked Loss")
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Epoch")
    ax.set_zlabel("Loss")

    ax = fig.add_subplot(122, projection="3d")
    X = np.arange(len(chunked_accuracy))
    Y = np.arange(chunk_size)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(chunked_accuracy).T  # Transpose the array to match the shape of X and Y
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_title("3D Surface Plot of Chunked Accuracy")
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Epoch")
    ax.set_zlabel("Accuracy")

    # Function to calculate the average of chunks
    def chunked_average(values, chunk_size):
        return [np.mean(values[i : i + chunk_size]) for i in range(0, len(values), chunk_size)]

    avg_accuracy_chunks = chunked_average(history["val_accuracy"], chunk_size)
    avg_loss_chunks = chunked_average(history["val_loss"], chunk_size)

    # Find the chunk with the highest average accuracy
    max_acc_chunk_index = np.argmax(avg_accuracy_chunks)
    max_acc_value = avg_accuracy_chunks[max_acc_chunk_index]

    # Create a pile plot for accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(avg_accuracy_chunks)),
        avg_accuracy_chunks,
        color="blue",
        label="Average Accuracy",
    )
    plt.bar(
        max_acc_chunk_index,
        max_acc_value,
        color="red",
        label="Highest Average Accuracy",
    )
    plt.xlabel("Chunk")
    plt.ylabel("Average Accuracy")
    plt.title("Average Validation Accuracy per Chunk")
    plt.legend()

    # Create a pile plot for loss
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(avg_loss_chunks)),
        avg_loss_chunks,
        color="green",
        label="Average Loss",
    )
    plt.xlabel("Chunk")
    plt.ylabel("Average Loss")
    plt.title("Average Validation Loss per Chunk")
    plt.legend()

    # Function to calculate the average of each epoch across chunks, ignoring the first chunk
    def average_across_chunks(values, chunk_size):
        num_chunks = len(values) // chunk_size
        avg_values = []
        for epoch in range(chunk_size):
            epoch_values = [values[chunk * chunk_size + epoch] for chunk in range(1, num_chunks)]
            avg_values.append(np.mean(epoch_values))
        return avg_values

    # Calculate the average accuracy and loss for each epoch across chunks, ignoring the first chunk
    avg_accuracy_epochs = average_across_chunks(history["val_accuracy"], chunk_size)
    avg_loss_epochs = average_across_chunks(history["val_loss"], chunk_size)

    # Create a bar plot for average accuracy and loss of each epoch across chunks
    plt.figure(figsize=(12, 6))

    # Create an index for each epoch
    epoch_indices = np.arange(len(avg_accuracy_epochs))

    # Plot accuracy and loss as bars
    plt.bar(
        epoch_indices - 0.2,
        avg_accuracy_epochs,
        width=0.4,
        label="Average Accuracy",
        color="blue",
        alpha=0.6,
    )
    plt.bar(
        epoch_indices + 0.2,
        avg_loss_epochs,
        width=0.4,
        label="Average Loss",
        color="orange",
        alpha=0.6,
    )

    # Add labels and title
    plt.xlabel("Epoch (within chunk)")
    plt.ylabel("Average Value")
    plt.title("Average Validation Accuracy and Loss for Each Epoch Across Chunks (Ignoring First Chunk)")
    plt.xticks(epoch_indices, [f"Epoch {i + 1}" for i in epoch_indices])  # Set x-tick labels to epoch numbers
    plt.legend()

    plt.tight_layout()
    plt.show()

except (ValueError, NameError) as E:
    print(f"\033[91mFailed to load model history.\nError: {E}")
