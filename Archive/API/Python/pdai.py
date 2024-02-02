import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from typing import Union, Dict
import numpy as np
import cv2

class PneumoniaModel:
    def __init__(self, model_path: str, verbose: int = 0):
        """
        Initializes the PneumoniaModel with the given model path and verbosity level.

        Args:
            model_path (str): Path to the saved model.
            verbose (int, optional): Verbosity level. If 1, prints status messages during operations. Defaults to 0.
        """
        self.model_path = model_path
        self.model = None
        self.verbose = verbose

    
    def load_model(self) -> Dict[str, Union[str, None]]:
        """
        Loads the model from the path specified during initialization.

        Returns:
            dict: A dictionary with a "status" key. If the model is loaded successfully, "status" is "success". 
                  If an error occurs, "status" is "error" and an additional "message" key contains the error message.
        """
        try:
            self.model = None 
            self.model = load_model(self.model_path)
            if self.verbose == 1:
                print("Model loaded successfully.")
        except Exception as e:
            if self.verbose == 1:
                print(f"Error loading model: {str(e)}")
            return {"status": "error", "message": str(e)}

        return {"status": "success"}


    def predict(self, image: np.ndarray, clahe: bool = False) -> Dict[str, Union[str, float, None]]:
        """
        Makes a prediction using the loaded model on the given image.

        Args:
            image (np.ndarray): The image to make a prediction on.
            clahe (bool, optional): Whether to apply CLAHE to the image before making a prediction. Defaults to False.

        Returns:
            dict: A dictionary with a "status" key. If the prediction is made successfully, "status" is "success", 
                  and additional "prediction" and "confidence" keys contain the prediction and confidence level. 
                  If an error occurs, "status" is "error" and an additional "message" key contains the error message.
        """
        if self.model is None:
            if self.verbose == 1:
                print("Model not loaded. Call load_model() first.")
            return {"status": "error", "message": "Model not loaded. Call load_model() first."}
        
        if image.ndim != 4 or image.shape[3] != 3:
            return {"status": "error", "message": f"Invalid image format. The image should have three color channels (RGB). Img shape = {image.shape}."}

        try:
            if clahe:
                # Create a CLAHE object
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
                
                b, g, r = cv2.split(image[0])
                
                # Convert the channels to the appropriate format
                b = cv2.convertScaleAbs(b)
                g = cv2.convertScaleAbs(g)
                r = cv2.convertScaleAbs(r)
                
                # Apply adaptive histogram equalization to each channel
                equalized_b = clahe.apply(b)
                equalized_g = clahe.apply(g)
                equalized_r = clahe.apply(r)

                # Merge the equalized channels back into an image
                equalized_image = cv2.merge((equalized_b, equalized_g, equalized_r))

                # Replace the original image with the equalized image in the array
                image = equalized_image

            # Normalize the image
            image = image / 255.0

            if self.verbose == 1:
                print("Making prediction...")
            prediction = self.model.predict(image)
            if np.argmax(prediction) == 0:
                if self.verbose == 1:
                    print("Prediction: Normal")
                return {"status": "success", "prediction": "Normal", "confidence": np.max(prediction)}
            else:
                if self.verbose == 1:
                    print("Prediction: Pneumonia")
                return {"status": "success", "prediction": "Pneumonia", "confidence": np.max(prediction)}
        except IndexError as e:
            if self.verbose == 1:
                print(f"Error making prediction: {str(e)}")
            return {"status": "error", "message": str(e)}
