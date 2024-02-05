# Pneumonia Prediction AI
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"/>

This project is an AI-based solution designed to predict pneumonia from X-ray images. The AI model processes the images at a resolution of 280x280 using a modified EfficientNetB7 with a custom classifier layer. It uses data augmentation to generate 28,000 samples for training and has achieved an accuracy of 95.51%.

The project is divided into two parts: 

1. **AI Training**: This part is responsible for training the AI model.
2. **CLI**: A colorful command-line interface (CLI) for using the trained AI model.

## Table of Contents

- [Pneumonia Prediction AI](#pneumonia-prediction-ai)
  - [Table of Contents](#table-of-contents)
  - [Releases](#releases)
  - [Usage](#usage)
  - [About the AI](#about-the-ai)
  - [Computing Environment](#computing-environment)
  - [Main Training File](#main-training-file)
  - [Cloning the Repository](#cloning-the-repository)
  - [Using the CLI Template](#using-the-cli-template)
  - [Contributing](#contributing)
  - [License](#license)

## Releases

There are two releases for this project:

1. **Source Release**: This release includes the source code for both the AI training and the CLI.
2. **CLI and Model Release**: This release includes only the CLI and the trained model for usage.

## Usage

You can run the CLI by executing the `CLI.cmd` file.

## About the AI

The AI is designed to predict pneumonia from X-ray images. It processes the images at a resolution of 280x280. The model is a modified EfficientNetB7 with a custom classifier layer. 

The model is implemented using Keras and TensorFlow, two of the most popular libraries for deep learning. Keras provides a high-level, user-friendly API for developing and training machine learning models.

To enhance the training, the model uses data augmentation to generate 28,000 samples for training. This technique helps improve the model's performance by providing a larger and more varied dataset for training.

The model has achieved an accuracy of 95.51% at predicting pneumonia from X-ray images.


## Computing Environment

The AI model was trained on a Windows machine with the following specifications:

- Processor: Intel Core i7-12700KF
- RAM: 64GB
- Graphics Card: NVIDIA GeForce RTX 3090

## Main Training File

The main file for training the AI model is `Model_TT.ipynb`. This Jupyter notebook contains all the code for training the model and should be used as the starting point for understanding the training process.

## Cloning the Repository

To clone the repository and run the project on your local machine, follow these steps:

1. Open your terminal.
2. Change the current working directory to the location where you want the cloned directory.
3. Type `git clone`, and then paste the URL of this repository. It will look something like this:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
```

4. Press Enter to create your local clone.

## Using the CLI Template

If you want to use the CLI for your own project, we provide a template version of the CLI. You can find it in the `cli-template` directory in this repository.

To use the CLI template, you need to replace the AI model with your own model and adjust the input and output processing to match your project's requirements.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
