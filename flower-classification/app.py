from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os

# Get path to images folder
dirname = os.path.dirname(__file__)
images_folder = os.path.join(dirname, 'images/Test')

# Create variables for your project
publish_iteration_name = "Iteration1"
project_id = "<YOUR_PROJECT_ID>"

# Create variables for your prediction resource
prediction_key = "<YOUR_KEY>"
endpoint = "<YOUR_ENDPOINT>"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)

print("Testing one image...\n")
# Open an image and make a prediction
with open(os.path.join(images_folder, "tigerlily4.jpg"), "rb") as image_contents:
    results = predictor.classify_image(project_id, publish_iteration_name, image_contents.read())

# Display the results
for prediction in results.predictions:
    print(f"{prediction.tag_name}: {prediction.probability * 100 :.2f}%")

print("\nTesting all the images...\n")
# Classify all the images
images = os.listdir(images_folder)
for i in range(len(images)):
    # Open the image, and use the custom vision model to classify it
    image_contents = open(os.path.join(images_folder, images[i]), "rb")
    results = predictor.classify_image(project_id, publish_iteration_name, image_contents.read())
    
    # Print the predicted class
    print(f"Image {images[i]}: {results.predictions[0].tag_name} {results.predictions[0].probability * 100 :.2f}%")