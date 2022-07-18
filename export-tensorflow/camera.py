import cv2
from predict import detect_image

# Set the size of the image (in pixels)
img_width = 640
img_height = 480

print("\n", "*"*8, "Starting camera!", "*"*8, "\n")
# Take an image from the camera and save it
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

ret, image = camera.read()
cv2.imwrite('capture.png', image)

results = detect_image('capture.png')
print("\n", "*"*8, "Captured image saved!", "*"*8, "\n")
print("\n", "*"*8, "Results", "*"*8, "\n")

# Select color for the bounding box (BGR)
colors = {
    "tomato": (0,215,255),
    "cucumber": (255,215,0),
    "pepper": (66,174,255)
}

# Display the results
for prediction in results:
    if prediction['probability'] > 0.3:
        print(f"{prediction['tagName']}: {prediction['probability'] * 100 :.2f}%")
        color = colors[prediction['tagName']]
        left = prediction['boundingBox']['left'] * img_width
        top = prediction['boundingBox']['top'] * img_height
        height = prediction['boundingBox']['height'] * img_height
        width =  prediction['boundingBox']['width'] * img_width
        result_image = cv2.rectangle(image, (int(left), int(top)), (int(left + width), int(top + height)), color, 3)
        cv2.putText(result_image, f"{prediction['tagName']}: {prediction['probability'] * 100 :.2f}%", (int(left), int(top)-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = color, thickness = 2)
        cv2.imwrite('result.png', result_image)

camera.release()