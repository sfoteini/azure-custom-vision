from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Create variables for your project
publish_iteration_name = "Iteration1"
project_id = "<PROJECT_ID>"

# Create variables for your prediction resource
prediction_key = "<YOUR_KEY>"
endpoint = "<YOUR_ENDPOINT>"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)

# Load a test image and get its dimensions
img_file = os.path.join('Images', 'Test', 'SantaClaus (2).jpg')
img = Image.open(img_file)
img_height, img_width, img_ch = np.array(img).shape

# Detect objects in the test image
with open(img_file, mode="rb") as test_img:
    results = predictor.detect_image(project_id, publish_iteration_name, test_img)

# Display the image
draw = ImageDraw.Draw(img)

# Select line width and color for the bounding box
lineWidth = int(img_width/100)
color = (0,255,0)

# Display the results
for prediction in results.predictions:
    if prediction.probability > 0.5:
        left = prediction.bounding_box.left * img_width
        top = prediction.bounding_box.top * img_height
        height = prediction.bounding_box.height * img_height
        width =  prediction.bounding_box.width * img_width
        # Create a rectangle
        draw.rectangle((left, top, left+width, top+height), outline=color, width=lineWidth)
        # Display probabilities
        font = ImageFont.truetype("arial.ttf", 18)
        draw.text((left, top-20), f"{prediction.probability * 100 :.2f}%", fill=color, font=font)
img.save("result.jpg")
print("Image saved!")