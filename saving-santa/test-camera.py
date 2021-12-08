import cv2
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Create variables for your project
publish_iteration_name = "Iteration1"
project_id = "<PROJECT_ID>"

# Create variables for your prediction resource
prediction_key = "<YOUR_KEY>"
endpoint = "<YOUR_ENDPOINT>"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)

# Set the size of the image (in pixels)
img_width = 640
img_height = 480

# Take an image from the camera and save it
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

ret, image = camera.read()
cv2.imwrite('capture.png', image)

# Call the prediction API
with open("capture.png", mode="rb") as captured_image:
    results = predictor.detect_image(project_id, publish_iteration_name, captured_image)

# Select color for the bounding box
color = (0,255,0)

# Display the results
for prediction in results.predictions:
    if prediction.probability > 0.5:
        left = prediction.bounding_box.left * img_width
        top = prediction.bounding_box.top * img_height
        height = prediction.bounding_box.height * img_height
        width =  prediction.bounding_box.width * img_width
        result_image = cv2.rectangle(image, (int(left), int(top)), (int(left + width), int(top + height)), color, 3)
        cv2.putText(result_image, f"{prediction.probability * 100 :.2f}%", (int(left), int(top)-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, color = color, thickness = 2)
        cv2.imwrite('result.png', result_image)
        print("Santa Claus detected! Image saved!")

camera.release()