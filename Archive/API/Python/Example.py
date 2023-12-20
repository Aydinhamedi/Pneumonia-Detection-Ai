# Import necessary libraries
import pprint
import numpy as np
from pdai import *
from PIL import Image
pp = pprint.PrettyPrinter(indent=4, width=10)

# Instantiate the PneumoniaModel class
pdai_model = PneumoniaModel("models\Ready\V1\PAI_model.h5", verbose=0)

# Load the model
pdai_model.load_model()

# Load an image for prediction
img_path = 'API\\Python\\test sampels\\PNEUMONIA\\person1947_bacteria_4876.jpeg'
img = Image.open(img_path)
img = img.convert('RGB')  # Convert grayscale to RGB
img = img.resize((280, 300))
x = np.array(img)
x = np.expand_dims(x, axis=0)

print('without CLAHE>>>')
# Make a prediction without CLAHE
result = pdai_model.predict(x)
pp.pprint(result)
print('with CLAHE>>>')
# Make a prediction with CLAHE
result = pdai_model.predict(x, clahe=True)
pp.pprint(result)
