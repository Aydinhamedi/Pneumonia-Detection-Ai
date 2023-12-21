# Pneumonia Detection AI
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"/>

### This project uses a deep learning model built with the TensorFlow framework to detect pneumonia in X-ray images. The model architecture is based on the EfficientNetB7 model, which has achieved an accuracy of approximately 97% (96.96%) on our test data. This high accuracy rate is one of the strengths of our AI model.
> [!NOTE]
>  Please note that this code uses my Python-CLI-template\
>  for more info go to https://github.com/Aydinhamedi/Python-CLI-template.

> [!NOTE]
>  Please note that this code uses my Python-color-print-V2\
>  for more info go to https://github.com/Aydinhamedi/Python-color-print-V2.

> [!NOTE]
>  Please note that this code uses my Python-color-print\
>  for more info go to https://github.com/Aydinhamedi/Python-color-print.
## Usage

The project includes a Command Line Interface (CLI) for easy use of the model. The CLI, which is based on the [Python CLI template](https://github.com/Aydinhamedi/Python-CLI-template) from the same author, provides a user-friendly, colorful interface that allows you to interact with the model. you can fined the cli in 

```
Interface\CLI
```
### Example Image of the CLI:
![Example](doc/Screenshot.png)  
## Release
> ### Newest release 📃
> #### [Go to newest release](https://github.com/Aydinhamedi/Ai-MNIST-Advanced-model/releases/tag/V0.3.6)

## Model

The model is a Convolutional Neural Network (CNN) trained on a dataset of 8888 X-ray images. The dataset is a combination of the [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and the [Covid19-Pneumonia-Normal Chest X-Ray Images](https://data.mendeley.com/datasets/dvntn9yhd2/1) from Mendeley.


## Training

The model provides two training methods: `rev1` and `rev2`.

- `rev1` is a simple Keras fit method.
- `rev2` uses a subset training method with augmentation.

The `rev2` method works as follows:

1. **Garbage Collection**: It starts by clearing the memory using Python's garbage collection and TensorFlow's backend clear session to free up RAM and GPU memory.

2. **Hyperparameters**: It sets up various hyperparameters for the training process. These include the maximum number of epochs, the number of epochs to train each subset, the size of each training subset, the batch size, and learning rates.

3. **Data Augmentation**: It uses Keras's ImageDataGenerator to augment the training data. This includes horizontal and vertical flips, rotation, zooming, shearing, width and height shifts, brightness adjustments, and channel shifts.

4. **Subset Training**: It trains the model on subsets of the training data. This is done to prevent overfitting and to make the training process more manageable. The size of the subsets and the number of epochs to train each subset are defined by the hyperparameters.

5. **Learning Rate Scheduling**: It uses a learning rate schedule to adjust the learning rate during training. The learning rate starts at a maximum value and decays over time to a minimum value.

6. **Callbacks**: It uses various Keras callbacks during training. These include early stopping to prevent overfitting, model checkpoint to save the best model, and TensorBoard for visualization.

7. **Training Loop**: It runs a training loop where it trains the model on each subset for a certain number of epochs. After each epoch, it checks the performance of the model on the validation data and saves the model if it has improved.

8. **Normalization**: It normalizes the image data to a range of 0 to 1. This is done to make the training process more stable and to improve the performance of the model.

9. **Noise Addition**: It adds random noise to the image data to increase the robustness of the model. The intensity of the noise is randomly chosen for each image.

Please note that the actual code may contain additional details and functionalities. This is just a high-level overview of how the `rev2` training method works.

## Repository Structure

Please note that due to the large size of some files and folders, they are not available directly in the repository. However, they can be found in the [Releases](https://github.com/Aydinhamedi/Pneumonia-Detection-Ai/releases) page of the repository. This includes the model weights and the database, which are crucial for the functioning of the AI model.

## Contribution

Any contributions to improve the project are welcome. You can submit a pull request or open an issue on GitHub. Please make sure to test your changes thoroughly before submitting. We appreciate your help in making this project better.

## WARNING
> [!CAUTION]
The model provided in this project should not be used for medical diagnosis without further validation. While the model has shown high accuracy in detecting pneumonia from X-ray images, it is not a substitute for professional medical advice. Please consult with a healthcare professional for medical advice.

## License

This project is open-source and is licensed under the MIT License. See the `LICENSE` file for details.

[GitHub Repository](https://github.com/Aydinhamedi/Pneumonia-Detection-Ai)
