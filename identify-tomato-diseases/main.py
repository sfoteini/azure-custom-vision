from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os, time

# Load the endpoint and keys of your resource
load_dotenv()
training_endpoint = os.getenv('TRAINING_ENDPOINT')
training_key = os.getenv('TRAINING_KEY')
prediction_endpoint = os.getenv('PREDICTION_ENDPOINT')
prediction_key = os.getenv('PREDICTION_KEY')
prediction_resource_id = os.getenv('PREDICTION_RESOURCE_ID')

# Authenticate the client
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(training_endpoint, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)

# Find the domain id
classification_domain = next(domain for domain in trainer.get_domains() if domain.type == "Classification" and domain.name == "General (compact)")

# Create a new project
publish_iteration_name = "Iteration1"
project_name = "Tomato leaf diseases"
project_description = "A Custom Vision project to classify tomato leaf diseases"
domain_id = classification_domain.id
classification_type = "Multiclass"
print ("Creating project...")
project = trainer.create_project(project_name, project_description, domain_id, classification_type)

# Add helper variables
tags_folder_names = [ "Bacterial_spot", "Early_blight", "Healthy", "Late_blight", "Leaf_Mold", 
                      "Septoria_leaf_spot", "Spider_mites", "Target_Spot", "Tomato_mosaic_virus",
                      "Tomato_Yellow_Leaf_Curl_Virus" ]
tags_description = [ "Bacterial spot", "Early blight", "Healthy", "Late blight", "Leaf Mold", 
                      "Septoria leaf spot", "Spider mites", "Target Spot", "Tomato mosaic virus",
                      "Tomato Yellow Leaf Curl Virus" ]

# Add tags
tags = [trainer.create_tag(project.id, tag_description) for tag_description in tags_description]

# Upload and tag images
images_folder = os.path.join(os.path.dirname(__file__), "images", "Train")

print("Adding images...")

for i in range(0, 10):
    image_list = []
    for image_num in range(1, 61):
        file_name = f"{tags_folder_names[i]} ({image_num}).JPG"
        with open(os.path.join(images_folder, tags_folder_names[i], file_name), "rb") as image_contents:
            image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[tags[i].id]))

    upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)
        exit(-1)
    print(f"{tags_folder_names[i]} Uploaded")

# Training
print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    print ("Waiting 10 seconds...")
    time.sleep(10)

# Get iteration performance information
threshold = 0.5
iter_performance_info = trainer.get_iteration_performance(project.id, iteration.id, threshold)
print("Iteration Performance: ")
print(f"\tPrecision: {iter_performance_info.precision*100 :.2f}%\n"
      f"\tRecall: {iter_performance_info.recall*100 :.2f}%\n"
      f"\tRecall: {iter_performance_info.average_precision*100 :.2f}%")

# Publish the current iteration
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Iteration published!")

# Test - Make a prediction
test_images_folder_path = os.path.join(os.path.dirname(__file__), "images", "Test")

print("Testing the prediction endpoint...")
test_image_filename = "Bacterial_spot (1).JPG"
with open(os.path.join(test_images_folder_path, test_image_filename), "rb") as image_contents:
    results = predictor.classify_image(project.id, publish_iteration_name, image_contents.read())

    # Display the results
    print(f"Testing image {test_image_filename}...")
    for prediction in results.predictions:
        print(f"\t{prediction.tag_name}: {prediction.probability*100 :.2f}%")